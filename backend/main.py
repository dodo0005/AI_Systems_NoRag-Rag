from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_PERCENT,
    DOCS_DIR,
    collection,
    delete_document,
    index_document,
    list_documents,
    rebuild_index,
)
from backend.rag import answer_question
from backend.schemas import AskRequest, AskResponse

app = FastAPI(
    title="NoRAG RAG API",
    description="Document retrieval and grounded question answering API",
    version="1.0.0",
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


@app.get("/api/documents")
def get_documents():
    """List supported documents and their indexing status."""
    try:
        return {"documents": list_documents()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not list documents.",
        ) from exc


@app.post("/api/documents/{document_id}/index")
def index_one_document(
    document_id: str,
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Index or reindex one document."""
    try:
        count = index_document(
            document_id,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
        )
        return {
            "status": "ok",
            "document_id": document_id,
            "chunk_count": count,
            "chunk_size": chunk_size,
            "overlap_percent": overlap_percent,
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
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Rebuild the index for every supported document in docs/."""
    try:
        result = rebuild_index(
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
        )

        if result["failures"]:
            result["status"] = "partial"
        else:
            result["status"] = "ok"

        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Index rebuild failed.",
        ) from exc


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=100, le=10000),
    overlap_percent: int = Query(DEFAULT_OVERLAP_PERCENT, ge=0, le=80),
):
    """Upload, validate, save, and index a .txt or .md file."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    # Normalize both Unix and Windows path separators.
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

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = (DOCS_DIR / filename).resolve()

    if destination.parent != DOCS_DIR.resolve():
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists. Delete it first if you want to replace it.",
        )

    # Read at most the size limit plus one byte.
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

    try:
        destination.write_text(decoded, encoding="utf-8")

        count = index_document(
            filename,
            chunk_size=chunk_size,
            overlap_percent=overlap_percent,
        )

        return {
            "status": "ok",
            "document_id": filename,
            "chunk_count": count,
            "chunk_size": chunk_size,
            "overlap_percent": overlap_percent,
        }

    except Exception as exc:
        # Avoid leaving an uploaded file or partial index behind
        # when indexing fails.
        try:
            collection.delete(where={"document_id": filename})
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
def remove_document(document_id: str):
    """Delete the source file and its indexed chunks."""
    try:
        delete_document(document_id)
        return {
            "status": "ok",
            "document_id": document_id,
            "message": "Document and indexed chunks deleted.",
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


@app.post("/api/ask", response_model=AskResponse)
def ask_question_endpoint(request: AskRequest):
    """Answer a question using only the selected documents."""
    try:
        return answer_question(
            question=request.question,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        # Invalid model output or invalid retrieval input.
        raise HTTPException(
            status_code=502,
            detail="The answer could not be validated. Please try again.",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="The answer service is not configured correctly.",
        ) from exc

    except Exception as exc:
        # Includes upstream provider/network errors.
        raise HTTPException(
            status_code=502,
            detail="The answer provider failed to process the question.",
        ) from exc
