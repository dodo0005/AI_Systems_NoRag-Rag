
import { useEffect, useState } from "react";
import "./App.css";

const API = "/api";

// Retrievers the user can switch on or off. RRF fuses the ones that are on.
const RETRIEVERS = [
  { id: "dense", label: "Dense (embeddings)" },
  { id: "bm25", label: "BM25 (keywords)" },
  { id: "graph", label: "Concept graph", corpora: ["kestrel"] },
];

export default function App() {
  const [corpus, setCorpus] = useState("vellora");
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [retrievers, setRetrievers] = useState(["dense", "bm25"]);
  const [chunkSize, setChunkSize] = useState(800);
  const [overlap, setOverlap] = useState(10);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [activeTab, setActiveTab] = useState("ask");
  const [wordMode, setWordMode] = useState("corpus");
  const [wordQuestion, setWordQuestion] = useState("");
  const [wordAnalysis, setWordAnalysis] = useState(null);
  const [wordLoading, setWordLoading] = useState(false);
  const [wordError, setWordError] = useState("");
  const [wordSearch, setWordSearch] = useState("");

  useEffect(() => {
  if (activeTab === "words" && wordMode === "corpus") {
    loadWordAnalysis("corpus");
  }
}, [activeTab, wordMode, corpus]);

  async function request(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    return data;
  }

  function corpusQuery(extra = {}) {
    const params = new URLSearchParams({
      corpus,
      ...extra,
    });

    return params.toString();
  }

  function documentId(doc) {
    return doc.document_id || doc.filename || doc.name;
  }



  async function loadWordAnalysis(mode, question = "") {
  setWordLoading(true);
  setWordError("");

  try {
    const params = new URLSearchParams({
      corpus,
      mode,
    });

    if (mode === "question") {
      params.set("question", question);
    }

    const response = await fetch(`${API}/words?${params}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Word analysis failed.");
    }

    const data = await response.json();
    setWordAnalysis(data);
  } catch (error) {
    setWordError(error.message || "Unable to load word analysis.");
    setWordAnalysis(null);
  } finally {
    setWordLoading(false);
  }
}

function analyzeQuestionWords(event) {
  event.preventDefault();

  if (!wordQuestion.trim()) {
    setWordError("Enter a question to analyze.");
    return;
  }

  loadWordAnalysis("question", wordQuestion.trim());
}

  async function loadDocuments() {
    try {
      const data = await request(
        `/documents?${corpusQuery()}`
      );

      const docs = Array.isArray(data)
        ? data
        : data.documents || data.items || [];

      setDocuments(docs);
      setSelected((previous) =>
        previous.filter((id) =>
          docs.some((doc) => documentId(doc) === id)
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
    setSelected([]);
    setResult(null);
    setDocuments([]);
    setError("");
    setNotice("");
    setFile(null);

    loadDocuments();
  }, [corpus]);

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
        `/documents?${corpusQuery({
          chunk_size: chunkSize,
          overlap_percent: overlap,
        })}`,
        { method: "POST", body: form }
      );

      setNotice(`Uploaded and indexed ${file.name} in ${corpus}.`);
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
        `/documents/${encodeURIComponent(id)}/index?${corpusQuery({
          chunk_size: chunkSize,
          overlap_percent: overlap,
        })}`,
        { method: "POST" }
      );

      setNotice(`Indexed ${id} in ${corpus}.`);
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
        `/index/rebuild?${corpusQuery({
          chunk_size: chunkSize,
          overlap_percent: overlap,
        })}`,
        { method: "POST" }
      );

      setNotice(
        `${corpus} index rebuilt: ${data.indexed_documents} documents, ` +
        `${data.indexed_chunks} chunks, ${Object.keys(data.failures || {}).length} failures.`
      );

      await loadDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function deleteDocument(id) {
    if (!window.confirm(`Delete "${id}" from ${corpus} and remove its indexed chunks?`)) {
      return;
    }

    setLoading(true);
    setError("");
    setNotice("");

    try {
      await request(
        `/documents/${encodeURIComponent(id)}?${corpusQuery()}`,
        { method: "DELETE" }
      );

      setSelected((previous) => previous.filter((item) => item !== id));
      setNotice(`Deleted ${id} from ${corpus}.`);
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

    if (activeRetrievers.length === 0) {
      setError("Switch on at least one retriever.");
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
          corpus,
          retrievers: activeRetrievers,
        }),
      });

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // A retriever with a "corpora" list exists only for those corpora.
  function isAvailable(retriever) {
    return !retriever.corpora || retriever.corpora.includes(corpus);
  }

  const activeRetrievers = retrievers.filter((id) =>
    isAvailable(RETRIEVERS.find((retriever) => retriever.id === id)),
  );

  function toggleRetriever(id) {
    setRetrievers((current) =>
      current.includes(id)
        ? current.filter((name) => name !== id)
        : [...current, id],
    );
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

            <button
              className="button secondary"
              onClick={loadDocuments}
              disabled={loading}
            >
              Refresh
            </button>
          </div>

          <div className="corpus-selector">
            <label className="field-label" htmlFor="corpus">
              Active knowledge base
            </label>

            <select
              id="corpus"
              value={corpus}
              onChange={(event) => setCorpus(event.target.value)}
              disabled={loading}
            >
              <option value="vellora">Vellora — Coffee machine support</option>
              <option value="kestrel">Kestrel — Billing and support</option>
            </select>

            <p className="helper-text">
              All document operations and questions use this knowledge base.
            </p>
          </div>

          <form className="upload-form" onSubmit={uploadDocument}>
            <label className="field-label" htmlFor="file">
              Add a document to {corpus}
            </label>

            <input
              id="file"
              type="file"
              accept=".txt,.md,text/plain,text/markdown"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />

            <button
              className="button primary full"
              disabled={loading || !file}
            >
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
                        {doc.indexed ? " · Indexed" : " · Not indexed"}
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
                No documents found in {corpus}. Upload a .txt or .md file to get started.
              </p>
            )}
          </div>

          <div className="settings">
            <p className="eyebrow">INDEX SETTINGS</p>

            <label className="setting-row">
              <span>Chunk size</span>
              <input
                type="number"
                min="100"
                max="10000"
                value={chunkSize}
                onChange={(event) => setChunkSize(event.target.value)}
              />
            </label>

            <label className="setting-row">
              <span>Overlap (%)</span>
              <input
                type="number"
                min="0"
                max="80"
                value={overlap}
                onChange={(event) => setOverlap(event.target.value)}
              />
            </label>

            <button
              className="button secondary full"
              onClick={rebuildIndex}
              disabled={loading}
            >
              Rebuild {corpus} index
            </button>

            <p className="helper-text">
              Rebuilding clears and recreates only the selected knowledge base index.
            </p>
          </div>
        </aside>

        <section className="main-column">
  <div className="panel tab-panel">
    <div className="tab-buttons">
      <button
        className={`tab-button ${activeTab === "ask" ? "selected" : ""}`}
        onClick={() => setActiveTab("ask")}
        type="button"
      >
        Ask documents
      </button>

      <button
        className={`tab-button ${activeTab === "words" ? "selected" : ""}`}
        onClick={() => setActiveTab("words")}
        type="button"
      >
        Words
      </button>
    </div>
  </div>

  {activeTab === "ask" && (
    <>
      <section className="panel ask-panel">
        <p className="eyebrow">RETRIEVAL AUGMENTED GENERATION</p>
        <h2>Ask your documents</h2>

        <p className="muted">
          Answers are generated from the selected documents, with source
          passages you can inspect.
        </p>

        <div className="active-corpus">
          Active knowledge base: <strong>{corpus}</strong>
        </div>

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

          <fieldset className="retriever-toggles">
            <legend>Retrievers (fused with RRF)</legend>

            {RETRIEVERS.map((retriever) => (
              <label key={retriever.id} className="retriever-toggle">
                <input
                  type="checkbox"
                  disabled={!isAvailable(retriever)}
                  checked={
                    isAvailable(retriever) && retrievers.includes(retriever.id)
                  }
                  onChange={() => toggleRetriever(retriever.id)}
                />
                {retriever.label}
                {retriever.corpora && (
                  <small className="retriever-note">
                    ({retriever.corpora.join(", ")} only)
                  </small>
                )}
              </label>
            ))}
          </fieldset>

          <div className="ask-controls">
            <label className="setting-row compact">
              <span>Retrieved chunks (top_k)</span>
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
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
              <p className="eyebrow">RESPONSE · {corpus.toUpperCase()}</p>
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

          <p className="answer-meta">
            Retrievers: {result.retrievers?.join(" + ") || "—"} ·{" "}
            {result.usage?.llm_calls ?? 0} LLM call
            {result.usage?.llm_calls === 1 ? "" : "s"} ·{" "}
            {(result.usage?.prompt_tokens ?? 0) +
              (result.usage?.completion_tokens ?? 0)}{" "}
            tokens ·{" "}
            {Math.round(
              Object.values(result.latency_ms || {}).reduce((a, b) => a + b, 0),
            )}{" "}
            ms
          </p>

          {result.graph_path && (
            <details className="graph-path" open>
              <summary>
                Graph path · {result.graph_path.entities.length} entit
                {result.graph_path.entities.length === 1 ? "y" : "ies"} in the
                question · {result.graph_path.edges.length} edges followed ·{" "}
                {result.graph_path.in_context.length}/
                {result.graph_path.chunk_ids.length} graph chunks reached the
                answer context
              </summary>

              <p>
                <strong>Entities found in the question:</strong>{" "}
                {result.graph_path.entities
                  .map((entity) => `${entity.name} (${entity.type})`)
                  .join(", ") || "none"}
              </p>

              <ol className="graph-edges">
                {result.graph_path.edges.map((edge, index) => (
                  <li key={index}>
                    <span className="hop">hop {edge.hop}</span> {edge.source}{" "}
                    <em>—{edge.relation}→</em> {edge.target}
                    <small>
                      {" "}
                      · “{edge.quote}” · chunk{" "}
                      {edge.chunk_ids
                        .map(
                          (id) =>
                            id.slice(0, 8) +
                            (result.graph_path.in_context.includes(id)
                              ? " (in context)"
                              : ""),
                        )
                        .join(", ")}
                    </small>
                  </li>
                ))}
              </ol>
            </details>
          )}

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
                      <small className="found-by">
                        Chunk {ref.chunk_id.slice(0, 8)} · fused rank #
                        {ref.fused_rank ?? "?"} · found by{" "}
                        {Object.entries(ref.found_by || {})
                          .map(([name, rank]) => `${name} #${rank}`)
                          .join(", ") || "—"}
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
    </>
  )}

  {activeTab === "words" && (
    <>
      <section className="panel words-panel">
        <p className="eyebrow">CORPUS ANALYSIS</p>
        <h2>Words explorer</h2>

        <p className="muted">
          Explore the vocabulary of the selected knowledge base or check
          which words from a question occur in its indexed chunks.
        </p>

        <div className="active-corpus">
          Active knowledge base: <strong>{corpus}</strong>
        </div>

        <div className="word-mode-buttons">
          <button
            type="button"
            className={`button ${wordMode === "corpus" ? "primary" : "secondary"}`}
            onClick={() => {
              setWordMode("corpus");
              setWordAnalysis(null);
              setWordSearch("");
            }}
          >
            Corpus mode
          </button>

          <button
            type="button"
            className={`button ${wordMode === "question" ? "primary" : "secondary"}`}
            onClick={() => {
              setWordMode("question");
              setWordAnalysis(null);
              setWordError("");
            }}
          >
            Question mode
          </button>
        </div>

        {wordMode === "corpus" && (
          <>
            <label className="field-label" htmlFor="word-search">
              Filter vocabulary
            </label>

            <input
              id="word-search"
              className="word-search"
              type="text"
              value={wordSearch}
              onChange={(event) => setWordSearch(event.target.value)}
              placeholder="Search for a word..."
            />
          </>
        )}

        {wordMode === "question" && (
          <form onSubmit={analyzeQuestionWords}>
            <label className="field-label" htmlFor="word-question">
              Question to analyze
            </label>

            <textarea
              id="word-question"
              rows="4"
              value={wordQuestion}
              onChange={(event) => setWordQuestion(event.target.value)}
              placeholder="Enter a question to inspect its vocabulary..."
            />

            <button
              type="submit"
              className="button primary"
              disabled={wordLoading}
            >
              Analyze question
            </button>
          </form>
        )}

        {wordLoading && (
          <p className="empty-state">Analyzing vocabulary...</p>
        )}

        {wordError && (
          <div className="alert error">{wordError}</div>
        )}
      </section>

      {wordAnalysis && wordMode === "corpus" && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h2>Corpus vocabulary</h2>
              <p className="muted">
                {wordAnalysis.total_unique_words} unique words
              </p>
            </div>
          </div>

          <div className="word-results">
            {wordAnalysis.words
              .filter((item) =>
                item.word.includes(wordSearch.trim().toLowerCase())
              )
              .map((item) => (
                <details className="word-entry" key={item.word}>
                  <summary>
                    <strong>{item.word}</strong>
                    <span>{item.frequency} occurrences</span>
                  </summary>

                  <div className="word-locations">
                    {item.chunks.map((chunk) => (
                      <div
                        className="word-location"
                        key={chunk.chunk_id}
                      >
                        <span>{chunk.document_id}</span>
                        <small>Chunk {chunk.chunk_index + 1}</small>
                      </div>
                    ))}
                  </div>
                </details>
              ))}
          </div>
        </section>
      )}

      {wordAnalysis && wordMode === "question" && (
        <section className="panel">
          <h2>Question vocabulary results</h2>

          <p className="muted">
            Words found in the selected corpus and the chunks containing them.
          </p>

          <div className="word-result-section">
            <h3>
              Found words ({wordAnalysis.found.length})
            </h3>

            {wordAnalysis.found.length > 0 ? (
              wordAnalysis.found.map((item) => (
                <details className="word-entry" key={item.word}>
                  <summary>
                    <strong>{item.word}</strong>
                    <span>{item.frequency} occurrences</span>
                  </summary>

                  <div className="word-locations">
                    {item.chunks.map((chunk) => (
                      <div
                        className="word-location"
                        key={chunk.chunk_id}
                      >
                        <span>{chunk.document_id}</span>
                        <small>Chunk {chunk.chunk_index + 1}</small>
                      </div>
                    ))}
                  </div>
                </details>
              ))
            ) : (
              <p className="empty-state">No meaningful words were found.</p>
            )}
          </div>

          <div className="word-result-section">
            <h3>
              Absent words ({wordAnalysis.absent.length})
            </h3>

            {wordAnalysis.absent.length > 0 ? (
              <div className="absent-words">
                {wordAnalysis.absent.map((word) => (
                  <span className="absent-word" key={word}>
                    {word}
                  </span>
                ))}
              </div>
            ) : (
              <p className="empty-state">
                All meaningful question words occur in the corpus.
              </p>
            )}
          </div>
        </section>
      )}
    </>
  )}

  <footer className="footer">
    Norag RAG · Evidence-based answers from your documents
  </footer>
</section>
      </div>
    </main>
  );
}
