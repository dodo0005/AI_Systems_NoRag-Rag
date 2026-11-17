import { useEffect, useMemo, useRef, useState } from "react";
import { request } from "./api";

const TYPE_COLORS = [
  "#344b34", "#8a5a2b", "#2d5d7b", "#7b3b5a", "#5b6b2b", "#6b4f9a", "#a0522d", "#3b7b6b",
];
const MAX_NEIGHBOURS = 18;

function colorOf(type, types) {
  return TYPE_COLORS[Math.max(0, types.indexOf(type)) % TYPE_COLORS.length];
}

// Radial layout: the selected entity in the centre, its neighbours on a circle.
function Neighbourhood({ graph, nodeId, onSelect, types }) {
  const node = graph.byId[nodeId];
  const links = (graph.adjacency[nodeId] || []).slice(0, MAX_NEIGHBOURS);
  const size = 460;
  const centre = size / 2;
  const radius = 170;

  const placed = links.map((edge, index) => {
    const other = edge.source === nodeId ? edge.target : edge.source;
    const angle = (2 * Math.PI * index) / Math.max(1, links.length) - Math.PI / 2;
    return {
      edge,
      other: graph.byId[other],
      x: centre + radius * Math.cos(angle),
      y: centre + radius * Math.sin(angle),
      outgoing: edge.source === nodeId,
    };
  });

  return (
    <svg
      className="graph-svg"
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Neighbourhood of ${node.name}`}
    >
      {placed.map(({ edge, x, y, outgoing }) => (
        <g key={edge.id}>
          <line x1={centre} y1={centre} x2={x} y2={y} className="graph-link" />
          <text
            x={(centre + x) / 2}
            y={(centre + y) / 2}
            className="graph-link-label"
            textAnchor="middle"
          >
            {outgoing ? `${edge.relation} →` : `← ${edge.relation}`}
          </text>
        </g>
      ))}
      {placed.map(({ edge, other, x, y }) => (
        <g
          key={`n-${edge.id}`}
          className="graph-node"
          onClick={() => onSelect(other.id)}
          tabIndex="0"
          onKeyDown={(event) => event.key === "Enter" && onSelect(other.id)}
        >
          <circle cx={x} cy={y} r="9" fill={colorOf(other.type, types)} />
          <text x={x} y={y + 22} textAnchor="middle">
            {other.name.length > 22 ? `${other.name.slice(0, 21)}…` : other.name}
          </text>
        </g>
      ))}
      <circle cx={centre} cy={centre} r="16" fill={colorOf(node.type, types)} />
      <text x={centre} y={centre + 32} textAnchor="middle" className="graph-centre-label">
        {node.name}
      </text>
    </svg>
  );
}

export default function GraphExplorer({ corpus, selected, onAsk, onGraphChange }) {
  const [data, setData] = useState(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState(null);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const poller = useRef(null);

  async function loadGraph() {
    try {
      const graph = await request(`/graph?corpus=${corpus}`);
      setData(graph);
      setMissing(false);
      setError("");
    } catch (err) {
      setData(null);
      setMissing(true);
      if (!/No concept graph/.test(err.message)) setError(err.message);
    }
  }

  async function pollStatus() {
    try {
      const s = await request(`/graph/status?corpus=${corpus}`);
      setStatus(s);
      if (s.state === "running") {
        poller.current = setTimeout(pollStatus, 1500);
      } else if (s.state === "done") {
        await loadGraph();
        onGraphChange?.();
      }
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    setCurrent(null);
    setHistory([]);
    setStatus(null);
    loadGraph();
    pollStatus();
    return () => clearTimeout(poller.current);
  }, [corpus]);

  async function build() {
    setError("");
    try {
      await request("/graph/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corpus, document_ids: selected }),
      });
      pollStatus();
    } catch (err) {
      setError(err.message);
    }
  }

  const graph = useMemo(() => {
    if (!data) return null;
    const byId = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
    const adjacency = {};
    for (const edge of data.edges) {
      (adjacency[edge.source] ||= []).push(edge);
      (adjacency[edge.target] ||= []).push(edge);
    }
    const types = [...new Set(data.nodes.map((n) => n.type))].sort();
    return { byId, adjacency, types };
  }, [data]);

  function select(id) {
    if (current) setHistory((h) => [...h.slice(-9), current]);
    setCurrent(id);
  }

  const matches = useMemo(() => {
    if (!data) return [];
    const needle = search.trim().toLowerCase();
    return data.nodes
      .filter((n) => !typeFilter || n.type === typeFilter)
      .filter(
        (n) =>
          !needle ||
          n.name.toLowerCase().includes(needle) ||
          (n.aliases || []).some((a) => a.toLowerCase().includes(needle)),
      )
      .sort(
        (a, b) =>
          (graph.adjacency[b.id]?.length || 0) - (graph.adjacency[a.id]?.length || 0) ||
          a.name.localeCompare(b.name),
      )
      .slice(0, 40);
  }, [data, search, typeFilter, graph]);

  const meta = data?.metadata || {};
  const node = current && graph?.byId[current];

  return (
    <section className="panel graph-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">CONCEPT MAP EXPLORER · {corpus.toUpperCase()}</p>
          <h2>Concept graph</h2>
        </div>
        <button
          className="button primary"
          type="button"
          onClick={build}
          disabled={status?.state === "running"}
        >
          {status?.state === "running"
            ? "Building…"
            : `${data ? "Rebuild" : "Build"} graph from ${
                selected.length ? `${selected.length} selected document(s)` : "all documents"
              }`}
        </button>
      </div>

      <p className="helper-text">
        One LLM call per chunk, cached: a rebuild only pays for new or changed chunks.
        Kestrel uses its own schema; any other document uses a general one (people,
        organizations, products, components, concepts, places, events).
      </p>

      {status && status.state !== "idle" && (
        <div className={`alert ${status.state === "failed" ? "error" : "success"}`}>
          {status.state === "running" && (
            <progress value={status.done || 0} max={status.total || 1} />
          )}{" "}
          {status.message}
        </div>
      )}
      {error && <div className="alert error">{error}</div>}

      {missing && !status?.state?.match(/running/) && (
        <p className="empty-state">
          No graph for {corpus} yet. Upload or select documents, then build it.
        </p>
      )}

      {data && graph && (
        <>
          <p className="graph-meta">
            {meta.counts?.nodes} entities · {meta.counts?.edges} relations · built{" "}
            {meta.built_at ? new Date(meta.built_at).toLocaleString() : "—"} with{" "}
            {meta.model} · schema {data.schema?.name || "kestrel"}
            {meta.documents && ` · from ${meta.documents.join(", ")}`}
          </p>

          <div className="graph-layout">
            <div className="graph-sidebar">
              <input
                type="search"
                placeholder="Find an entity…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
              <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
                <option value="">All types</option>
                {graph.types.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
              <ul className="entity-list">
                {matches.map((n) => (
                  <li key={n.id}>
                    <button
                      type="button"
                      className={n.id === current ? "selected" : ""}
                      onClick={() => select(n.id)}
                    >
                      <span
                        className="type-dot"
                        style={{ background: colorOf(n.type, graph.types) }}
                      />
                      {n.name}
                      <small>
                        {n.type} · {graph.adjacency[n.id]?.length || 0}
                      </small>
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            <div className="graph-main">
              {!node && (
                <p className="empty-state">
                  Pick an entity: the most connected ones are listed first.
                </p>
              )}

              {node && (
                <>
                  <div className="graph-node-header">
                    <div>
                      <strong>{node.name}</strong> <small>({node.type})</small>
                      {node.aliases?.length > 0 && (
                        <small> · also: {node.aliases.join(", ")}</small>
                      )}
                    </div>
                    <div className="inline-controls">
                      {history.length > 0 && (
                        <button
                          className="text-button"
                          type="button"
                          onClick={() => {
                            setCurrent(history[history.length - 1]);
                            setHistory((h) => h.slice(0, -1));
                          }}
                        >
                          ← Back
                        </button>
                      )}
                      <button
                        className="button secondary"
                        type="button"
                        onClick={() => onAsk(node.name)}
                      >
                        Ask about it
                      </button>
                    </div>
                  </div>

                  <Neighbourhood
                    graph={graph}
                    nodeId={node.id}
                    onSelect={select}
                    types={graph.types}
                  />
                  {(graph.adjacency[node.id]?.length || 0) > MAX_NEIGHBOURS && (
                    <p className="helper-text">
                      Showing {MAX_NEIGHBOURS} of {graph.adjacency[node.id].length} relations
                      in the picture; all are listed below.
                    </p>
                  )}

                  <ol className="graph-edges">
                    {(graph.adjacency[node.id] || []).map((edge) => (
                      <li key={edge.id}>
                        <button
                          type="button"
                          className="link-button"
                          onClick={() => select(edge.source)}
                        >
                          {graph.byId[edge.source].name}
                        </button>{" "}
                        <em>—{edge.relation}→</em>{" "}
                        <button
                          type="button"
                          className="link-button"
                          onClick={() => select(edge.target)}
                        >
                          {graph.byId[edge.target].name}
                        </button>
                        {edge.evidence.map((ev) => (
                          <small key={ev.chunk_id}>
                            <br />“{ev.quote}” · {ev.document_id} · chunk {ev.chunk_id.slice(0, 8)}
                          </small>
                        ))}
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
