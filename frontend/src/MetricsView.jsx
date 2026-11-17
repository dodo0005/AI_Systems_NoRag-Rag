import { useEffect, useState } from "react";
import { formatMs, formatUsd, request } from "./api";

const OUTCOMES = ["answered", "abstained", "invalid", "error", "blocked"];

export default function MetricsView() {
  const [days, setDays] = useState(30);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      setMetrics(await request(`/metrics?days=${days}`));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load();
  }, [days]);

  if (error) return <div className="alert error">{error}</div>;
  if (!metrics) return <p className="muted">Loading metrics…</p>;

  const q = metrics.questions;
  const evaluated = metrics.last_evaluation_fingerprints || [];
  const stale = metrics.fingerprints.filter((f) => !evaluated.includes(f));

  return (
    <section className="panel metrics-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">OBSERVABILITY · COMPUTED FROM THE TRACES</p>
          <h2>Metrics</h2>
        </div>
        <div className="inline-controls">
          <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
            <option value={1}>Last 24 h</option>
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
          </select>
          <button className="button secondary" onClick={load} type="button">
            Refresh
          </button>
        </div>
      </div>

      <div className="stat-grid">
        <div className="stat">
          <small>Questions</small>
          <strong>{q.total}</strong>
        </div>
        {OUTCOMES.map((name) => (
          <div className="stat" key={name}>
            <small>{name}</small>
            <strong>{q.outcomes[name] || 0}</strong>
            <small>{q.total ? Math.round(((q.outcomes[name] || 0) / q.total) * 100) : 0}%</small>
          </div>
        ))}
      </div>

      <div className="stat-grid">
        <div className="stat">
          <small>Request p50</small>
          <strong>{formatMs(metrics.latency_ms.request.p50)}</strong>
        </div>
        <div className="stat">
          <small>Request p95</small>
          <strong>{formatMs(metrics.latency_ms.request.p95)}</strong>
        </div>
        <div className="stat">
          <small>Tokens / question</small>
          <strong>{metrics.per_question.avg_tokens ?? "—"}</strong>
        </div>
        <div className="stat">
          <small>Cost / question</small>
          <strong>{formatUsd(metrics.per_question.avg_cost_usd)}</strong>
        </div>
        <div className="stat">
          <small>LLM calls / question</small>
          <strong>{metrics.per_question.avg_llm_calls ?? "—"}</strong>
        </div>
      </div>

      {stale.length > 0 && (
        <div className="alert warning">
          Configuration {stale.join(", ")} answered questions but is not in the last
          evaluation run ({evaluated.join(", ") || "none"}): its quality was never measured.
        </div>
      )}

      <h3>Latency per step</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Step (span)</th>
            <th>p50</th>
            <th>p95</th>
            <th>n</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics.latency_ms.steps).map(([name, v]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{formatMs(v.p50)}</td>
              <td>{formatMs(v.p95)}</td>
              <td>{v.n}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Per indexed document</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Document</th>
            <th>Index runs</th>
            <th>Chunks</th>
            <th>LLM calls (graph)</th>
            <th>Tokens</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics.per_document).map(([doc, v]) => (
            <tr key={doc}>
              <td>{doc}</td>
              <td>{v.index_runs}</td>
              <td>{v.chunks}</td>
              <td>{v.llm_calls}</td>
              <td>{v.tokens}</td>
              <td>{formatUsd(v.cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="helper-text">
        Embeddings are computed locally, so indexing costs nothing; only graph extraction calls the LLM.
      </p>

      <h3>Per day</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Day (UTC)</th>
            <th>Questions</th>
            <th>Index runs</th>
            <th>LLM calls</th>
            <th>Tokens</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics.per_day).map(([day, v]) => (
            <tr key={day}>
              <td>{day}</td>
              <td>{v.questions}</td>
              <td>{v.index_runs}</td>
              <td>{v.llm_calls}</td>
              <td>{v.tokens}</td>
              <td>{formatUsd(v.cost_usd)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
