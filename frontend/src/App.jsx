
import { useEffect, useState } from "react";
import "./App.css";

const API = "/api";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [chunkSize, setChunkSize] = useState(800);
  const [overlap, setOverlap] = useState(10);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [apiOnline, setApiOnline] = useState(false);

  async function request(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }
    return data;
  }

  async function loadDocuments() {
    try {
      const data = await request("/documents");
      const docs = Array.isArray(data)
        ? data
        : data.documents || data.items || [];

      setDocuments(docs);
      setSelected((previous) =>
        previous.filter((id) =>
          docs.some((doc) => (doc.document_id || doc.filename || doc.name) === id)
        )
      );
      setApiOnline(true);
      setError("");
    } catch (err) {
      setApiOnline(false);
      setError(`Could not load documents: ${err.message}`);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  function documentId(doc) {
    return doc.document_id || doc.filename || doc.name;
  }

  function toggleDocument(id) {
    setSelected((previous) =>
      previous.includes(id)
        ? previous.filter((item) => item !== id)
        : [...previous, id]
    );
  }

  async function uploadDocument(event) {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", file);

      await request(
        `/documents?chunk_size=${chunkSize}&overlap_percent=${overlap}`,
        { method: "POST", body: form }
      );

      setNotice(`Uploaded and indexed ${file.name}.`);
      setFile(null);
      event.target.reset();
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function indexDocument(id) {
    setLoading(true);
    setError("");
    setNotice("");

    try {
      await request(
        `/documents/${encodeURIComponent(id)}/index?chunk_size=${chunkSize}&overlap_percent=${overlap}`,
        { method: "POST" }
      );
      setNotice(`Indexed ${id}.`);
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function rebuildIndex() {
    setLoading(true);
    setError("");
    setNotice("");

    try {
      const data = await request(
        `/index/rebuild?chunk_size=${chunkSize}&overlap_percent=${overlap}`,
        { method: "POST" }
      );
      setNotice(data.message || "Index rebuild completed.");
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function deleteDocument(id) {
    if (!window.confirm(`Delete "${id}" and its indexed chunks?`)) return;

    setLoading(true);
    setError("");
    setNotice("");

    try {
      await request(`/documents/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      setSelected((previous) => previous.filter((item) => item !== id));
      setNotice(`Deleted ${id}.`);
      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    setError("");
    setNotice("");
    setResult(null);

    if (!question.trim()) {
      setError("Enter a question first.");
      return;
    }
    if (selected.length === 0) {
      setError("Select at least one document.");
      return;
    }

    setLoading(true);

    try {
      const data = await request("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          document_ids: selected,
          top_k: Number(topK),
        }),
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AI SYSTEMS · DOCUMENT SEARCH</p>
          <h1>Norag RAG</h1>
          <p className="subtitle">
            Ask questions. Get answers grounded in your documents.
          </p>
        </div>
        <div className={`connection ${apiOnline ? "online" : "offline"}`}>
          <span className="status-dot" />
          {apiOnline ? "API connected" : "API unavailable"}
        </div>
      </header>

      {error && <div className="alert error">{error}</div>}
      {notice && <div className="alert success">{notice}</div>}

      <div className="workspace">
        <aside className="panel documents-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">YOUR KNOWLEDGE BASE</p>
              <h2>Documents</h2>
            </div>
            <button className="button secondary" onClick={loadDocuments}>
              Refresh
            </button>
          </div>

          <form className="upload-form" onSubmit={uploadDocument}>
            <label className="field-label" htmlFor="file">
              Add a document
            </label>
            <input
              id="file"
              type="file"
              accept=".txt,.md,text/plain,text/markdown"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <button className="button primary full" disabled={loading || !file}>
              Upload and index
            </button>
          </form>

          <div className="document-list-heading">
            <span className="field-label">Available documents</span>
            <span className="count">{documents.length}</span>
          </div>

          <div className="document-list">
            {documents.map((doc) => {
              const id = documentId(doc);
              if (!id) return null;

              return (
                <div className="document-item" key={id}>
                  <label className="document-select">
                    <input
                      type="checkbox"
                      checked={selected.includes(id)}
                      onChange={() => toggleDocument(id)}
                    />
                    <span className="file-icon">TXT</span>
                    <span className="document-name">
                      <strong>{id}</strong>
                      <small>
                        {doc.chunk_count ?? doc.chunks ?? 0} chunks
                      </small>
                    </span>
                  </label>
                  <div className="document-actions">
                    <button
                      className="text-button"
                      disabled={loading}
                      onClick={() => indexDocument(id)}
                      title="Index document"
                    >
                      Index
                    </button>
                    <button
                      className="text-button danger-text"
                      disabled={loading}
                      onClick={() => deleteDocument(id)}
                      title="Delete document"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}

            {documents.length === 0 && (
              <p className="empty-state">
                No documents found. Upload a .txt or .md file to get started.
              </p>
            )}
          </div>

          <div className="settings">
            <p className="eyebrow">INDEX SETTINGS</p>

            <label className="setting-row">
              <span>Chunk size</span>
              <input
                type="number"
                min="1"
                max="10000"
                value={chunkSize}
                onChange={(e) => setChunkSize(e.target.value)}
              />
            </label>

            <label className="setting-row">
              <span>Overlap (%)</span>
              <input
                type="number"
                min="0"
                max="80"
                value={overlap}
                onChange={(e) => setOverlap(e.target.value)}
              />
            </label>

            <button
              className="button secondary full"
              onClick={rebuildIndex}
              disabled={loading}
            >
              Rebuild all indexes
            </button>
            <p className="helper-text">
              Rebuilding clears and recreates the current index.
            </p>
          </div>
        </aside>

        <section className="main-column">
          <section className="panel ask-panel">
            <p className="eyebrow">RETRIEVAL AUGMENTED GENERATION</p>
            <h2>Ask your documents</h2>
            <p className="muted">
              Answers are generated from the selected documents, with source
              passages you can inspect.
            </p>

            <div className="selection-summary">
              {selected.length} document{selected.length === 1 ? "" : "s"} selected
            </div>

            <form onSubmit={askQuestion}>
              <label className="field-label" htmlFor="question">
                Your question
              </label>
              <textarea
                id="question"
                rows="4"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="e.g. What should I do when error E05 appears?"
              />

              <div className="ask-controls">
                <label className="setting-row compact">
                  <span>Retrieved chunks (top_k)</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                  />
                </label>

                <button
                  className="button primary ask-button"
                  disabled={loading || !apiOnline}
                >
                  {loading ? "Working..." : "Ask question"}
                </button>
              </div>
            </form>
          </section>

          {result && (
            <section className="panel answer-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">RESPONSE</p>
                  <h2>Your answer</h2>
                </div>
                <span
                  className={`answer-status ${
                    result.status === "answered" ? "answered" : "insufficient"
                  }`}
                >
                  {result.status === "answered"
                    ? "Answered"
                    : "Insufficient evidence"}
                </span>
              </div>

              <p className="answer-text">{result.answer}</p>

              <div className="references">
                <h3>Source references ({result.references?.length || 0})</h3>

                {result.references?.length > 0 ? (
                  result.references.map((ref) => (
                    <details className="reference" key={ref.chunk_id}>
                      <summary>
                        <span>
                          <strong>{ref.document_id}</strong>
                          <small>
                            Lines {ref.start_line}–{ref.end_line}
                          </small>
                        </span>
                        <span className="open-label">View passage +</span>
                      </summary>
                      <pre>{ref.text}</pre>
                    </details>
                  ))
                ) : (
                  <p className="empty-state">
                    No supporting references were returned.
                  </p>
                )}
              </div>
            </section>
          )}

          <footer className="footer">
            Norag RAG · Evidence-based answers from your documents
          </footer>
        </section>
      </div>
    </main>
  );
}
