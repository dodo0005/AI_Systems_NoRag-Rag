import { useEffect, useState } from "react";
import { formatMs, formatUsd, request } from "./api";

// Spans arrive flat with a parent_id: order them as a tree, depth first.
function spanTree(spans) {
  const children = {};
  for (const span of spans) {
    const parent = span.parent_id || "root";
    (children[parent] ||= []).push(span);
  }
  const rows = [];
  function walk(parent, depth) {
    for (const span of children[parent] || []) {
      rows.push({ ...span, depth });
      walk(span.span_id, depth + 1);
    }
  }
  walk("root", 0);
  return rows;
}

function TraceDetail({ traceId }) {
  const [trace, setTrace] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setTrace(null);
    setError("");
    request(`/traces/${traceId}`).then(setTrace).catch((err) => setError(err.message));
  }, [traceId]);

  if (error) return <div className="alert error">{error}</div>;
  if (!trace) return <p className="muted">Loading trace {traceId}…</p>;

  const total = trace.duration_ms || 1;
  const rows = spanTree(trace.spans || []);

  return (
    <div className="trace-detail">
      <div className="trace-summary">
        <span className={`outcome outcome-${trace.outcome}`}>{trace.outcome}</span>
        <strong>{trace.name}</strong>
        <small>
          {trace.kind} · {new Date(trace.started_at).toLocaleString()} ·{" "}
          {formatMs(trace.duration_ms)} · {trace.usage?.llm_calls ?? 0} LLM call(s) ·{" "}
          {(trace.usage?.input_tokens ?? 0) + (trace.usage?.output_tokens ?? 0)} tokens ·{" "}
          {formatUsd(trace.usage?.cost_usd)} · config {trace.fingerprint || "—"}
        </small>
      </div>

      {trace.attributes?.question && (
        <p className="trace-text">
          <strong>Question:</strong> {trace.attributes.question}
          {trace.attributes.rewritten_question && (
            <>
              <br />
              <strong>Rewritten:</strong> {trace.attributes.rewritten_question}
            </>
          )}
        </p>
      )}
      {trace.attributes?.answer && (
        <p className="trace-text">
          <strong>Answer:</strong> {trace.attributes.answer}
        </p>
      )}

      <div className="waterfall">
        {rows.map((span) => (
          <details key={span.span_id} className={`span-row status-${span.status}`}>
            <summary>
              <span className="span-name" style={{ paddingLeft: span.depth * 16 }}>
                {span.name}
              </span>
              <span className="span-bar-track">
                <span
                  className="span-bar"
                  style={{
                    left: `${(span.start_ms / total) * 100}%`,
                    width: `${Math.max(0.6, ((span.duration_ms || 0) / total) * 100)}%`,
                  }}
                />
              </span>
              <span className="span-ms">{formatMs(span.duration_ms)}</span>
            </summary>
            {span.error && <p className="span-error">{span.error}</p>}
            <pre>{JSON.stringify(span.attributes, null, 2)}</pre>
          </details>
        ))}
      </div>

      <details className="trace-config">
        <summary>Configuration ({trace.fingerprint})</summary>
        <pre>{JSON.stringify(trace.config, null, 2)}</pre>
        <pre>{JSON.stringify(trace.attributes, null, 2)}</pre>
      </details>
    </div>
  );
}

export default function TraceView({ traceId, onSelect }) {
  const [traces, setTraces] = useState([]);
  const [kind, setKind] = useState("");
  const [error, setError] = useState("");

  async function load() {
    try {
      const data = await request(`/traces?limit=100${kind ? `&kind=${kind}` : ""}`);
      setTraces(data.traces);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load();
  }, [kind, traceId]);

  return (
    <section className="panel traces-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">OBSERVABILITY</p>
          <h2>Traces</h2>
        </div>
        <div className="inline-controls">
          <select value={kind} onChange={(event) => setKind(event.target.value)}>
            <option value="">All</option>
            <option value="question">Questions</option>
            <option value="index">Indexing</option>
          </select>
          <button className="button secondary" onClick={load} type="button">
            Refresh
          </button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}

      {traceId && <TraceDetail traceId={traceId} />}

      <table className="data-table trace-list">
        <thead>
          <tr>
            <th>When</th>
            <th>Trace</th>
            <th>Outcome</th>
            <th>Duration</th>
            <th>Tokens</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {traces.map((t) => (
            <tr
              key={t.trace_id}
              className={t.trace_id === traceId ? "selected" : ""}
              onClick={() => onSelect(t.trace_id)}
            >
              <td>{new Date(t.started_at).toLocaleTimeString()}</td>
              <td>
                {t.name}
                {t.attributes?.question && <small> · {t.attributes.question.slice(0, 60)}</small>}
              </td>
              <td>
                <span className={`outcome outcome-${t.outcome}`}>{t.outcome}</span>
              </td>
              <td>{formatMs(t.duration_ms)}</td>
              <td>{(t.usage?.input_tokens ?? 0) + (t.usage?.output_tokens ?? 0)}</td>
              <td>{formatUsd(t.usage?.cost_usd)}</td>
            </tr>
          ))}
          {traces.length === 0 && (
            <tr>
              <td colSpan="6" className="empty-state">
                No traces yet. Ask a question or index a document.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </section>
  );
}
