"""Tracing: one trace per question and per indexing run, one span per step.

Vocabulary from OpenTelemetry (trace, span, parent, attributes, status), but no
SDK and no external service: traces are written to a local SQLite file,
traces/traces.db, and read back by the API (/api/traces, /api/metrics).

    with trace("question", "ask", config=snapshot) as tr:
        with span("retrieve") as s:
            s.set(chunk_ids=[...])
        tr.set(outcome="answered")

Outside a trace (the evaluation scripts, the unit tests), span() does nothing,
so the pipeline code can always call it.

What never goes in a trace: API keys or other secrets (scrubbed again on write,
whatever the caller passes), and chunk text (spans carry chunk ids only).
Questions and answers are stored AFTER personal-data redaction (backend/redact.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DB = Path(os.getenv("TRACE_DB", PROJECT_ROOT / "traces" / "traces.db"))
RETENTION_DAYS = float(os.getenv("TRACE_RETENTION_DAYS", "30"))

# Attribute keys that must never be stored, whatever the caller does.
FORBIDDEN_KEY_PARTS = ("api_key", "apikey", "secret", "password", "authorization")

_current_trace: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)
_current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)
_write_lock = threading.Lock()


def _now_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds")


def _secrets() -> list[str]:
    values = [os.getenv("GOOGLE_API_KEY", "")]
    values += os.getenv("GOOGLE_API_KEYS", "").split(",")
    return [v.strip() for v in values if v and len(v.strip()) >= 8]


def scrub(value, secrets: list[str] | None = None):
    """Drop forbidden keys and blank any secret value, recursively."""
    secrets = _secrets() if secrets is None else secrets

    if isinstance(value, dict):
        return {
            k: scrub(v, secrets)
            for k, v in value.items()
            if not any(part in str(k).lower() for part in FORBIDDEN_KEY_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[SECRET]")
        return value
    return value


class Span:
    def __init__(self, trace: "Trace", name: str, parent: "Span | None"):
        self.trace = trace
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.parent_id = parent.id if parent else None
        self.start = time.perf_counter()
        self.duration_ms: float | None = None
        self.status = "unset"  # OpenTelemetry: unset, ok, error
        self.error: str | None = None
        self.attributes: dict = {}

    def set(self, **attributes) -> "Span":
        self.attributes.update(attributes)
        return self

    def fail(self, error: str) -> "Span":
        self.status = "error"
        self.error = error
        return self

    def to_row(self) -> dict:
        return {
            "span_id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_ms": round((self.start - self.trace.start) * 1000, 1),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class _NoSpan:
    """Returned by span() outside a trace: accepts everything, records nothing."""

    id = None

    def set(self, **attributes):
        return self

    def fail(self, error: str):
        return self


class Trace:
    def __init__(self, kind: str, name: str, config: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind  # "question" or "index"
        self.name = name
        self.started_at = time.time()
        self.start = time.perf_counter()
        self.duration_ms: float | None = None
        self.config = config or {}
        self.fingerprint = config_fingerprint(self.config) if config else None
        self.attributes: dict = {}
        self.outcome: str | None = None
        self.spans: list[Span] = []
        # Totals over every LLM call in the trace (filled by backend/llm.py).
        self.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    def set(self, **attributes) -> "Trace":
        if "outcome" in attributes:
            self.outcome = attributes.pop("outcome")
        self.attributes.update(attributes)
        return self

    def add_usage(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.usage["cost_usd"] = round(self.usage["cost_usd"] + cost_usd, 8)


def config_fingerprint(config: dict) -> str:
    """Short hash of the configuration: same fingerprint = same pipeline (C9 §3)."""
    import hashlib

    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]


def current_trace() -> Trace | None:
    return _current_trace.get()


@contextmanager
def trace(kind: str, name: str, config: dict | None = None, **attributes):
    """Open a trace. Inside an open trace (an index run inside an upload), join it:
    the caller's own spans then land in the outer trace."""
    existing = _current_trace.get()
    if existing is not None:
        yield existing
        return

    tr = Trace(kind, name, config)
    tr.set(**attributes)
    trace_token = _current_trace.set(tr)
    span_token = _current_span.set(None)

    try:
        yield tr
    except BaseException as exc:
        if tr.outcome is None:
            tr.outcome = "error"
        tr.attributes.setdefault("error_type", type(exc).__name__)
        raise
    finally:
        tr.duration_ms = round((time.perf_counter() - tr.start) * 1000, 1)
        if tr.outcome is None:
            tr.outcome = "ok"
        _current_trace.reset(trace_token)
        _current_span.reset(span_token)
        try:
            save_trace(tr)
        except Exception as exc:  # tracing must never break the app
            print(f"[tracing] could not save trace {tr.id}: {exc}")


@contextmanager
def span(name: str, **attributes):
    """Open a span under the current span. A no-op outside a trace."""
    tr = _current_trace.get()
    if tr is None:
        yield _NoSpan()
        return

    sp = Span(tr, name, _current_span.get())
    sp.set(**attributes)
    tr.spans.append(sp)
    token = _current_span.set(sp)

    try:
        yield sp
        if sp.status == "unset":
            sp.status = "ok"
    except BaseException as exc:
        if sp.status != "error":
            sp.fail(f"{type(exc).__name__}: {str(exc)[:200]}")
        raise
    finally:
        sp.duration_ms = round((time.perf_counter() - sp.start) * 1000, 1)
        _current_span.reset(token)


# ---------- storage ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    started_at REAL NOT NULL,
    duration_ms REAL,
    outcome TEXT,
    fingerprint TEXT,
    config TEXT,
    attributes TEXT,
    usage TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_id TEXT,
    name TEXT NOT NULL,
    start_ms REAL,
    duration_ms REAL,
    status TEXT,
    error TEXT,
    attributes TEXT,
    PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at);
"""


def _connect() -> sqlite3.Connection:
    TRACE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_trace(tr: Trace) -> None:
    secrets = _secrets()
    with _write_lock, _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                tr.id,
                tr.kind,
                tr.name,
                tr.started_at,
                tr.duration_ms,
                tr.outcome,
                tr.fingerprint,
                json.dumps(scrub(tr.config, secrets), default=str),
                json.dumps(scrub(tr.attributes, secrets), default=str),
                json.dumps(tr.usage),
            ),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    tr.id,
                    row["span_id"],
                    row["parent_id"],
                    row["name"],
                    row["start_ms"],
                    row["duration_ms"],
                    row["status"],
                    scrub(row["error"], secrets),
                    json.dumps(scrub(row["attributes"], secrets), default=str),
                )
                for row in (sp.to_row() for sp in tr.spans)
            ],
        )
        purge_old(conn)


def purge_old(conn: sqlite3.Connection | None = None) -> int:
    """Retention: delete traces older than TRACE_RETENTION_DAYS."""
    cutoff = time.time() - RETENTION_DAYS * 86400
    own = conn is None
    conn = conn or _connect()
    try:
        old = [r[0] for r in conn.execute("SELECT trace_id FROM traces WHERE started_at < ?", (cutoff,))]
        if old:
            marks = ",".join("?" * len(old))
            conn.execute(f"DELETE FROM spans WHERE trace_id IN ({marks})", old)
            conn.execute(f"DELETE FROM traces WHERE trace_id IN ({marks})", old)
        if own:
            conn.commit()
        return len(old)
    finally:
        if own:
            conn.close()


def _trace_row(row: sqlite3.Row) -> dict:
    return {
        "trace_id": row["trace_id"],
        "kind": row["kind"],
        "name": row["name"],
        "started_at": _now_iso(row["started_at"]),
        "duration_ms": row["duration_ms"],
        "outcome": row["outcome"],
        "fingerprint": row["fingerprint"],
        "config": json.loads(row["config"] or "{}"),
        "attributes": json.loads(row["attributes"] or "{}"),
        "usage": json.loads(row["usage"] or "{}"),
    }


def list_traces(kind: str | None = None, limit: int = 50) -> list[dict]:
    with _connect() as conn:
        query = "SELECT * FROM traces"
        params: list = []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [_trace_row(r) for r in conn.execute(query, params)]


def get_trace(trace_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        result = _trace_row(row)
        spans = [
            {
                "span_id": s["span_id"],
                "parent_id": s["parent_id"],
                "name": s["name"],
                "start_ms": s["start_ms"],
                "duration_ms": s["duration_ms"],
                "status": s["status"],
                "error": s["error"],
                "attributes": json.loads(s["attributes"] or "{}"),
            }
            for s in conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ms, rowid", (trace_id,)
            )
        ]
    result["spans"] = spans
    return result


# ---------- metrics, computed from the traces only ----------


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the value below which p% of requests finish."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * p // 100))  # ceil(n * p / 100)
    return round(ordered[int(rank) - 1], 1)


def compute_metrics(days: int = 30) -> dict:
    cutoff = time.time() - days * 86400
    with _connect() as conn:
        traces = [_trace_row(r) for r in conn.execute(
            "SELECT * FROM traces WHERE started_at >= ? ORDER BY started_at", (cutoff,)
        )]
        ids = [t["trace_id"] for t in traces]
        spans_by_trace: dict[str, list[dict]] = defaultdict(list)
        if ids:
            marks = ",".join("?" * len(ids))
            for s in conn.execute(f"SELECT * FROM spans WHERE trace_id IN ({marks})", ids):
                spans_by_trace[s["trace_id"]].append(
                    {"name": s["name"], "duration_ms": s["duration_ms"],
                     "attributes": json.loads(s["attributes"] or "{}")}
                )

    questions = [t for t in traces if t["kind"] == "question"]
    indexing = [t for t in traces if t["kind"] == "index"]

    outcomes = defaultdict(int)
    for t in questions:
        outcomes[t["outcome"] or "unknown"] += 1

    # Latency: whole request, and each step (span name), p50 and p95.
    step_durations: dict[str, list[float]] = defaultdict(list)
    for t in questions:
        for s in spans_by_trace[t["trace_id"]]:
            if s["duration_ms"] is not None:
                step_durations[s["name"]].append(s["duration_ms"])
    totals = [t["duration_ms"] for t in questions if t["duration_ms"] is not None]

    # Tokens and cost per question.
    q_tokens = [t["usage"].get("input_tokens", 0) + t["usage"].get("output_tokens", 0) for t in questions]
    q_cost = [t["usage"].get("cost_usd", 0.0) for t in questions]

    # Per indexed document: chunk counts from index.document spans (embedding is
    # local, so 0 $), LLM tokens and cost from graph.extract spans.
    per_doc: dict[str, dict] = defaultdict(lambda: {"index_runs": 0, "chunks": 0, "llm_calls": 0,
                                                    "tokens": 0, "cost_usd": 0.0})
    for t in indexing:
        for s in spans_by_trace[t["trace_id"]]:
            a = s["attributes"]
            doc = a.get("document_id")
            if not doc:
                continue
            entry = per_doc[doc]
            if s["name"] == "index.document":
                entry["index_runs"] += 1
                entry["chunks"] += a.get("chunk_count", 0) or 0
            if s["name"] == "graph.extract":
                entry["llm_calls"] += a.get("llm_calls", 0) or 0
                entry["tokens"] += (a.get("tokens_in", 0) or 0) + (a.get("tokens_out", 0) or 0)
                entry["cost_usd"] = round(entry["cost_usd"] + (a.get("cost_usd", 0) or 0), 6)

    # Per day, everything that called the LLM (questions and indexing).
    per_day: dict[str, dict] = defaultdict(lambda: {"questions": 0, "index_runs": 0, "llm_calls": 0,
                                                    "tokens": 0, "cost_usd": 0.0})
    for t in traces:
        day = t["started_at"][:10]
        entry = per_day[day]
        entry["questions" if t["kind"] == "question" else "index_runs"] += 1
        entry["llm_calls"] += t["usage"].get("llm_calls", 0)
        entry["tokens"] += t["usage"].get("input_tokens", 0) + t["usage"].get("output_tokens", 0)
        entry["cost_usd"] = round(entry["cost_usd"] + t["usage"].get("cost_usd", 0.0), 6)

    return {
        "window_days": days,
        "questions": {
            "total": len(questions),
            "outcomes": dict(outcomes),
        },
        "latency_ms": {
            "request": {"p50": percentile(totals, 50), "p95": percentile(totals, 95), "n": len(totals)},
            "steps": {
                name: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
                for name, v in sorted(step_durations.items())
            },
        },
        "per_question": {
            "avg_tokens": round(statistics.mean(q_tokens), 1) if q_tokens else None,
            "avg_cost_usd": round(statistics.mean(q_cost), 6) if q_cost else None,
            "avg_llm_calls": round(statistics.mean([t["usage"].get("llm_calls", 0) for t in questions]), 2)
            if questions else None,
        },
        "per_document": dict(sorted(per_doc.items())),
        "per_day": dict(sorted(per_day.items(), reverse=True)),
        "fingerprints": sorted({t["fingerprint"] for t in questions if t["fingerprint"]}),
    }
