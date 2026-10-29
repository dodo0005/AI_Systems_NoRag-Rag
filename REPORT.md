# REPORT.md

## Team Details

- Team members:Mahmoud NASSER, Mohamad Ibaad SHAMS, Joseph AYOUB
- Work distribution: mahmoud is backend, mohamad ibaad shams is front end, and joseph ayoub is evaluation and testing and reporting 

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
- 1. Irrelevant chunks retrieved for unsupported questions

Question: "What should I do when error E00 appears?"

Expected behavior: The system should indicate that the corpus does not establish instructions for E00.

Observed behavior: The API returned insufficient_evidence, but retrieval still returned chunks about other error codes and unrelated manual content.

Impact: Irrelevant passages can enter the generation context even when the corpus does not contain evidence for the requested error code.

Current handling: The answer generator correctly returned an insufficient-evidence response in the observed test.

Potential improvement for V2: Improve retrieval filtering or relevance handling for queries with explicit error codes, and evaluate whether this reduces irrelevant context without losing relevant passages.

- 2. Retrieval evaluation coverage is limited

Question set: The initial and expanded tests cover E05, E00, and nine additional error-code questions.

Observed result: The initial run achieved 1/1 retrieval hits for answerable questions and 2/2 expected statuses. The expanded run achieved 9/9 retrieval hits and 9/9 expected statuses.

Limitation: Most tested questions ask directly about explicit error codes in a single manual. These results do not establish how well retrieval handles indirect wording, questions requiring multiple passages, or other document types.

Potential improvement: Expand the evaluation with harder questions, paraphrases, multi-passage questions, and additional unanswerable questions. Use the same dataset and metric when comparing V1 and V2.


### Work ownership

- Person 1 — Backend owner: FastAPI setup, RAG pipeline,
  indexing, retrieval, API schemas, and backend integration.
- Person 2 — Frontend owner: User interface, document
  management controls, question submission, and reference display.
- Person 3 — Evaluation owner: Evaluation questions, retrieval
  testing, failure testing, and report coordination.

### Implementation log

#### Step 1 — Person 1: Created the FastAPI application,
  health endpoint, CORS configuration, and initial API schemas.
- Validation: Pending local execution.


#### Step 2 — Document loading and indexing

- Owner: Person 1 (backend).
- Implemented UTF-8 document loading for `.txt` and `.md`.
- Implemented configurable character-based chunking and overlap.
- Stored source document IDs, chunk IDs, and line ranges in ChromaDB metadata.
- Added document listing, individual indexing, and full-index rebuild endpoints.
- Validation: Record actual test results here.
- Known limitations: Full-index rebuild currently clears the existing
  collection before indexing the replacement corpus. If indexing fails,
  the previous index may not be recoverable automatically.

#### Step 3 — Document management

- Owner: Person 1 (backend).
- Implemented upload for .txt and .md files.
- Added file size, format, UTF-8, and duplicate filename validation.
- Implemented deletion of the source file and its indexed chunks.
- Validation: Record the actual upload, duplicate, invalid-format,
  listing, and deletion test results.
- Known limitations: File-system and vector-store changes are not
  transactional; recovery from partial storage failures could be improved.


#### Step 4 — Retrieval and answer generation**

- **Owner:** Person 1 (backend).
- Implemented ChromaDB retrieval restricted to selected documents and Gemini-based answer generation using retrieved evidence.
- Added response validation, source references, and `insufficient_evidence` handling.
- Improved retrieval by prioritizing exact error-code matches.
- **Validation:** E05 returned the correct answer and reference; unsupported E00 returned `insufficient_evidence`. Empty questions, invalid `top_k`, and invalid document identifiers were tested.
- **Known limitations:** Character-based chunking produces approximate line ranges. Systematic evaluation of retrieval and answer quality remains pending.



#### Step 5 — Frontend setup**

- **Owner:** Person 2 (frontend).
- Created the React frontend project using Vite.
- Installed the frontend dependencies and verified that the development server starts successfully.
- **Validation:** Development server starts and serves the frontend locally.
- **Status:** Frontend implementation and integration with the backend API pending.


#### Step 6 — Initial evaluation setup and testing

- **Owner:** Person 3.
- Created an initial evaluation dataset in `evaluation/evaluation_set.csv`, covering one answerable question (E05) and one question without established supporting evidence (E00).
- Implemented `evaluation/evaluate.py` to test retrieval through the RAG pipeline, call the question-answering API, and save retrieved chunks, generated answers, and statuses to a CSV results file.
- Defined retrieval Hit@k as whether the expected source-text marker appears among the retrieved chunks for an answerable question.
- **Initial results:** With top-k = 5, E05 retrieved the expected source passage (Hit@5: 1/1, 100%). Both test cases returned their expected statuses (2/2, 100%): E05 returned `answered`, and E00 returned `insufficient_evidence`.
- **Observed limitation:** For E00, retrieval returned unrelated passages about other error codes, although the answer generator correctly abstained in this test.
- **Current status:** Initial evaluation completed. The evaluation dataset is being expanded, and the expanded run must be completed before drawing broader conclusions or comparing system versions.