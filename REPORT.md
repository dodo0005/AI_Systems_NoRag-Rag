# REPORT.md

## Team Details

- Team members:Mahmoud NASSER, Mohamad Ibaad SHAMS, Joseph AYOUB
- Work distribution: To be completed

## Decision Log

### D1 — Corpus
**Options:** One literary corpus; one technical manual; a mixed corpus.

**Choice:** Use *Crime and Punishment* and the Vellora Duo espresso-machine manual.

**Reason:** The novel tests retrieval across a large body of text. The manual tests precise factual questions involving quantities, settings, and procedures.

### D2 — Embedding model
**Options:** Local Sentence Transformers; hosted embedding API.

**Choice:** `all-MiniLM-L6-v2`.

**Reason:** It is already used in the project, runs locally, and avoids additional embedding API costs.

### D3 — Vector database
**Options:** ChromaDB; FAISS; a hosted vector database.

**Choice:** Persistent local ChromaDB.

**Reason:** Existing code can be reused, and persistence allows the index to survive application restarts.

### D4 — LLM provider
**Options:** Google Gemini; OpenAI API; another compatible provider.

**Choice:** Google Gemini through the OpenAI-compatible SDK.

**Reason:** The current implementation already uses this integration.

### D5 — Backend
**Options:** A standalone Python script; FastAPI; Flask.

**Choice:** FastAPI.

**Reason:** It provides an HTTP API and integrates with Pydantic response validation.

### D6 — Frontend
**Options:** Streamlit; HTML/CSS/JavaScript; a React application.

**Choice:** HTML/CSS/JavaScript served by the backend.

**Reason:** This keeps the frontend lightweight while supporting document controls, configurable parameters, and clickable references.

### D7 — V1 chunking
**Options:** Fixed-size chunks with overlap; paragraph-based chunking; semantic chunking.

**Choice:** Start with fixed-size character chunks and configurable overlap.

**Reason:** This preserves the existing implementation and establishes a simple baseline for later comparison.

### D8 — Evaluation
**Options:** Manual inspection; a fixed evaluation dataset with an automated retrieval metric.

**Choice:** A versioned question set with expected evidence passages and Recall@k evaluation.

**Reason:** Repeated evaluations can use the same questions and evidence definitions to compare V1 and V2.

## Known Failures

To be completed after running reproducible tests. For each failure, record the test question, expected behavior, actual observation, and whether the failure is in retrieval, generation, validation, or the user interface.
