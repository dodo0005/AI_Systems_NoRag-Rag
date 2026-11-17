import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend import llm, tracing
from backend.graph import graph_file
from backend.graph_build import GraphBudgetError, build_graph
from backend.rag import (
    CORPORA,
    PROJECT_ROOT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_PERCENT,
    DEFAULT_CORPUS,
    get_docs_dir,
    get_collection,
    delete_document,
    index_document,
    list_documents,
    rebuild_index,
    answer_question,
    analyze_words,
    RetrievalError,
)
from backend.scanner import as_dicts, is_high_risk, scan
from backend.schemas import AskRequest, AskResponse, GraphBuildRequest
from backend.tracing import span, trace

QUARANTINE_DIR = PROJECT_ROOT / "quarantine"

app = FastAPI(
    title="NoRAG RAG API",
    description="Document retrieval and grounded question answering API",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "norag-rag-backend",
    }


@app.get("/api/config")
def get_config():
    """What the UI needs to know: the LLM in use and which corpora have a graph."""
    return {
        **llm.describe(),
        "llm_configured": bool(llm.settings().keys),
        "corpora": {
            name: {"has_graph": graph_file(name).exists()} for name in CORPORA
        },
    }


@app.get("/api/documents")
def get_documents(corpus: str = Query(DEFAULT_CORPUS)):
    try:
        return {"documents": list_documents(corpus=corpus)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not list documents.",
        ) from exc


@app.post("/api/documents/{document_id}/index")
def index_one_document(
    document_id: str,
    corpus: str = Query(DEFAULT_CORPUS),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Index or reindex one document in the selected corpus."""
    try:
        count = index_document(
            document_id,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            corpus=corpus,
        )

        return {
            "status": "ok",
            "document_id": document_id,
            "chunk_count": count,
            "chunk_size": chunk_size,
            "overlap_percent": overlap_percent,
            "corpus": corpus,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed.",
        ) from exc


@app.post("/api/index/rebuild")
def rebuild_all_documents(
    corpus: str = Query(DEFAULT_CORPUS),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Rebuild the index for every supported document in one corpus."""
    try:
        result = rebuild_index(
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            corpus=corpus,
        )

        result["status"] = "partial" if result["failures"] else "ok"
        result["corpus"] = corpus
        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Index rebuild failed.",
        ) from exc


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    corpus: str = Query(DEFAULT_CORPUS),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Upload, validate, save, and index a .txt or .md file."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    # Normalize Unix and Windows path separators.
    filename = file.filename.replace("\\", "/").split("/")[-1]

    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).suffix.lower() not in {".txt", ".md"}
    ):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are supported.",
        )

    try:
        docs_dir = get_docs_dir(corpus)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    docs_dir.mkdir(parents=True, exist_ok=True)
    destination = (docs_dir / filename).resolve()

    # Ensure the uploaded file stays inside the selected corpus directory.
    if destination.parent != docs_dir.resolve():
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "A document with this filename already exists. "
                "Delete it first if you want to replace it."
            ),
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The maximum upload size is 10 MB.",
        )

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must contain valid UTF-8 text.",
        ) from exc

    if not decoded.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is empty.",
        )

    config = {"corpus": corpus, "chunk_size": chunk_size, "overlap_percent": overlap_percent}
    with trace("index", f"upload {filename}", config=config, corpus=corpus) as tr:
        # Injection scanner: a poisoned document never reaches the index.
        with span("guard.document", document_id=filename) as sp:
            findings = scan(decoded)
            sp.set(findings=as_dicts(findings), quarantined=is_high_risk(findings))

        if is_high_risk(findings):
            tr.set(outcome="quarantined", document_id=filename)
            quarantine(corpus, filename, decoded, findings)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{filename} was not indexed: it contains instructions aimed at the "
                    f"model ({', '.join(sorted({f.rule for f in findings if f.severity == 'high'}))}). "
                    f"It was moved to quarantine/{corpus}/ for review (trace {tr.id})."
                ),
                headers={"X-Trace-Id": tr.id},
            ) from None

        return _save_and_index(destination, decoded, filename, corpus, chunk_size,
                               overlap_percent, findings, tr)


def quarantine(corpus: str, filename: str, text: str, findings) -> dict:
    folder = QUARANTINE_DIR / corpus
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (folder / f"{stamp}-{filename}").write_text(text, encoding="utf-8")
    report = {"document_id": filename, "corpus": corpus, "quarantined_at": stamp,
              "findings": as_dicts(findings)}
    (folder / f"{stamp}-{filename}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _save_and_index(destination, decoded, filename, corpus, chunk_size, overlap_percent,
                    findings, tr):
    try:
        destination.write_text(decoded, encoding="utf-8")

        count = index_document(
            filename,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
            corpus=corpus,
        )
        tr.set(document_id=filename, chunk_count=count)

        return {
            "status": "ok",
            "document_id": filename,
            "chunk_count": count,
            "chunk_size": chunk_size,
            "overlap_percent": overlap_percent,
            "corpus": corpus,
            "trace_id": tr.id,
            # Medium findings do not block the upload, but the user sees them.
            "warnings": [f"{f.rule} (line {f.line}): {f.excerpt}" for f in findings],
        }

    except Exception as exc:
        # Clean up only this corpus's index and uploaded file.
        try:
            get_collection(corpus).delete(where={"document_id": filename})
        except Exception:
            pass

        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass

        if isinstance(exc, ValueError):
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="The file could not be indexed.",
        ) from exc


@app.delete("/api/documents/{document_id}")
def remove_document(
    document_id: str,
    corpus: str = Query(DEFAULT_CORPUS),
):
    """Delete the source file and its indexed chunks."""
    try:
        delete_document(document_id, corpus=corpus)

        return {
            "status": "ok",
            "document_id": document_id,
            "message": "Document and indexed chunks deleted.",
            "corpus": corpus,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
        ) from exc


@app.get("/api/words")
def get_words(
    corpus: str = Query(DEFAULT_CORPUS),
    mode: str = Query("corpus"),
    question: str = Query(""),
):
    """Return corpus vocabulary or analyse words in a question."""
    if mode not in {"corpus", "question"}:
        raise HTTPException(
            status_code=400,
            detail="Mode must be 'corpus' or 'question'.",
        )

    if mode == "question" and not question.strip():
        raise HTTPException(
            status_code=400,
            detail="A question is required in question mode.",
        )

    try:
        return analyze_words(
            corpus=corpus,
            question=question if mode == "question" else "",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Word analysis failed.",
        ) from exc


def _with_trace(message: str, exc: Exception) -> str:
    trace_id = getattr(exc, "trace_id", None)
    return f"{message} (trace {trace_id})" if trace_id else message


@app.post("/api/ask", response_model=AskResponse)
def ask_question_endpoint(request: AskRequest):
    """Answer a question using only the selected corpus and documents."""
    try:
        return answer_question(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
            corpus=request.corpus,
            retrievers=request.retrievers,
            history=[turn.model_dump() for turn in request.history],
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except RetrievalError as exc:
        # The request itself is wrong (document, retriever, missing graph).
        raise HTTPException(
            status_code=400,
            detail=_with_trace(str(exc), exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=_with_trace("The answer could not be validated. Please try again.", exc),
        ) from exc

    except llm.LLMRateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail=_with_trace("The LLM provider is rate-limiting us. Try again in a minute.", exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=_with_trace(
                "The answer service is not configured correctly (LLM_URL, LLM_API_KEY, LLM_MODEL).",
                exc,
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=_with_trace("The answer provider failed to process the question.", exc),
        ) from exc


# ---------- Observability: traces and metrics (Part A) ----------


@app.get("/api/traces")
def get_traces(kind: str | None = Query(None), limit: int = Query(50, ge=1, le=500)):
    if kind not in {None, "question", "index"}:
        raise HTTPException(status_code=400, detail="kind must be 'question' or 'index'.")
    return {"traces": tracing.list_traces(kind=kind, limit=limit)}


@app.get("/api/traces/{trace_id}")
def get_one_trace(trace_id: str):
    result = tracing.get_trace(trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trace not found (deleted after the retention period?).")
    return result


@app.get("/api/metrics")
def get_metrics(days: int = Query(30, ge=1, le=365)):
    metrics = tracing.compute_metrics(days=days)
    # Is the running configuration the one the last evaluation measured?
    metrics["last_evaluation_fingerprints"] = last_evaluation_fingerprints()
    return metrics


def last_evaluation_fingerprints() -> list[str]:
    import csv

    found: set[str] = set()
    for path in sorted((PROJECT_ROOT / "evaluation").glob("summary_v*.csv"))[-1:]:
        with path.open(encoding="utf-8") as file:
            found |= {row.get("fingerprint") for row in csv.DictReader(file) if row.get("fingerprint")}
    return sorted(found)


@app.get("/api/quarantine")
def get_quarantine(corpus: str = Query(DEFAULT_CORPUS)):
    if corpus not in CORPORA:
        raise HTTPException(status_code=400, detail="Unknown corpus.")
    folder = QUARANTINE_DIR / corpus
    reports = []
    if folder.exists():
        for path in sorted(folder.glob("*.json"), reverse=True):
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    return {"documents": reports}


# ---------- Concept graph of any corpus: build in the background, explore ----------

_graph_jobs: dict[str, dict] = {}
_graph_lock = threading.Lock()


@app.get("/api/graph")
def get_graph(corpus: str = Query(DEFAULT_CORPUS)):
    if corpus not in CORPORA:
        raise HTTPException(status_code=400, detail="Unknown corpus.")
    path = graph_file(corpus)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No concept graph for {corpus} yet.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "nodes": data["nodes"],
        "edges": data["edges"],
        "schema": data.get("schema", {}),
        "metadata": data.get("metadata", {}),
    }


@app.get("/api/graph/status")
def graph_status(corpus: str = Query(DEFAULT_CORPUS)):
    with _graph_lock:
        return _graph_jobs.get(corpus, {"state": "idle"})


@app.post("/api/graph/build", status_code=202)
def start_graph_build(request: GraphBuildRequest):
    corpus = request.corpus
    with _graph_lock:
        if _graph_jobs.get(corpus, {}).get("state") == "running":
            raise HTTPException(status_code=409, detail="A graph is already being built for this corpus.")
        _graph_jobs[corpus] = {"state": "running", "done": 0, "total": 0, "message": "Starting..."}

    def progress(done: int, total: int):
        with _graph_lock:
            _graph_jobs[corpus].update(done=done, total=total, message=f"Chunk {done}/{total}")

    def run():
        try:
            graph = build_graph(corpus, document_ids=request.document_ids or None, progress=progress)
            meta = graph["metadata"]
            state = {"state": "done", "message": (
                f"{meta['counts']['nodes']} entities, {meta['counts']['edges']} relations "
                f"from {meta['chunks']} chunks ({meta['cost']['llm_calls_this_run']} LLM calls)."),
                "trace_id": meta.get("trace_id")}
        except Exception as exc:  # shown to the user, the trace has the details
            state = {"state": "failed", "message": str(exc)}
        with _graph_lock:
            _graph_jobs[corpus].update(state)

    # The budget check is cheap: run it now, so the user gets a 400 at once.
    try:
        from backend.graph_build import Extractor, MAX_NEW_CHUNKS, load_chunks
        from backend.graph_schema import schema_for

        chunks = load_chunks(corpus)
        if request.document_ids:
            chunks = [c for c in chunks if c["document_id"] in set(request.document_ids)]
        if not chunks:
            raise ValueError("No indexed chunks for these documents. Index them first.")
        extractor = Extractor(corpus, schema_for(corpus))
        new = sum(1 for c in chunks if not extractor.is_cached(c))
        if MAX_NEW_CHUNKS and new > MAX_NEW_CHUNKS:
            raise GraphBudgetError(
                f"This graph needs {new} new LLM calls (one per chunk), more than the budget of "
                f"{MAX_NEW_CHUNKS} (GRAPH_MAX_NEW_CHUNKS). Select fewer or smaller documents."
            )
        if new and not llm.settings().keys:
            raise ValueError("LLM_API_KEY is not set: the graph cannot be extracted.")
    except ValueError as exc:
        with _graph_lock:
            _graph_jobs[corpus] = {"state": "failed", "message": str(exc)}
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _graph_lock:
        _graph_jobs[corpus].update(total=len(chunks), message=f"{new} new chunks to extract")
    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "corpus": corpus, "chunks": len(chunks), "new_chunks": new}
