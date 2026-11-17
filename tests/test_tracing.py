"""Part A: traces, spans, secrets never stored, metrics from traces only."""

from backend import tracing


def test_trace_with_nested_spans_is_saved_and_read_back():
    with tracing.trace("question", "ask", config={"model": "m"}) as tr:
        with tracing.span("retrieve") as outer:
            with tracing.span("retrieve.bm25") as inner:
                inner.set(chunk_ids=["a", "b"])
        tr.set(outcome="answered")

    saved = tracing.get_trace(tr.id)
    assert saved["outcome"] == "answered"
    assert saved["fingerprint"] == tracing.config_fingerprint({"model": "m"})
    names = {s["name"]: s for s in saved["spans"]}
    assert names["retrieve.bm25"]["parent_id"] == names["retrieve"]["span_id"]
    assert names["retrieve.bm25"]["attributes"]["chunk_ids"] == ["a", "b"]


def test_exception_marks_trace_and_span_as_error():
    try:
        with tracing.trace("question", "ask") as tr:
            with tracing.span("generate"):
                raise RuntimeError("provider down")
    except RuntimeError:
        pass
    saved = tracing.get_trace(tr.id)
    assert saved["outcome"] == "error"
    assert saved["spans"][0]["status"] == "error"


def test_secrets_are_scrubbed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-very-secret-value")
    with tracing.trace("question", "ask", config={"api_key": "x", "url": "u"}) as tr:
        tr.set(note="key is sk-very-secret-value")
    saved = tracing.get_trace(tr.id)
    assert "api_key" not in saved["config"]
    assert "sk-very-secret-value" not in str(saved)


def test_span_outside_a_trace_does_nothing():
    with tracing.span("retrieve") as sp:
        sp.set(anything=1)  # no error, nothing stored


def test_percentiles_nearest_rank():
    values = [1.5] * 94 + [20.0] * 6
    assert tracing.percentile(values, 50) == 1.5
    assert tracing.percentile(values, 95) == 20.0


def test_metrics_count_outcomes_and_steps():
    for outcome in ("answered", "answered", "abstained", "invalid"):
        with tracing.trace("question", "ask") as tr:
            with tracing.span("generate"):
                pass
            tr.add_usage(100, 10, 0.001)
            tr.set(outcome=outcome)
    m = tracing.compute_metrics()
    assert m["questions"]["outcomes"]["answered"] >= 2
    assert m["questions"]["outcomes"]["invalid"] >= 1
    assert "generate" in m["latency_ms"]["steps"]
    assert m["per_question"]["avg_tokens"] > 0
