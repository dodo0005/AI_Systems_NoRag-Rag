# NoRAG-RAG — AI Systems Integration & Operations

A multi-corpus Retrieval-Augmented Generation (RAG) application using FastAPI, React, ChromaDB, Sentence Transformers and any OpenAI-compatible LLM (Google Gemini by default).

## Features

- Two independent knowledge bases: **Vellora** and **Kestrel**.
- Document upload, listing, indexing, rebuilding and deletion.
- Persistent ChromaDB collections isolated by corpus.
- Configurable chunk size and overlap.
- Semantic retrieval with optional exact error-code matching.
- Gemini answer generation grounded in retrieved evidence.
- Source references and insufficient-evidence handling.
- Words analysis:
  - Corpus vocabulary, word frequencies and chunk locations.
  - Question vocabulary analysis showing present and absent words.
- Repeatable retrieval evaluation and saved answer-generation results.

## Project structure

```text
norag-rag/
├── backend/
│   ├── main.py
│   ├── rag.py
│   └── schemas.py
├── frontend/
├── docs/                         # Vellora documents
├── corpora/
│   └── kestrel/
│       ├── docs/                 # Kestrel documents
│       └── eval/
│           └── questions.starter.jsonl
├── evaluation/
│   ├── evaluate.py
│   ├── no_context.py
│   ├── questions.vellora.jsonl
│   ├── questions.kestrel.jsonl
│   ├── no_context_vellora.jsonl
│   ├── no_context_kestrel.jsonl
│   ├── no_context_grading_v2.csv
│   ├── results_v2_*.jsonl
│   ├── answers_v2_*.jsonl
│   ├── grading_v2.csv
│   └── summary_v2.csv
├── chroma_db/
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.10 or later.
- Node.js and npm.
- An API key for an OpenAI-compatible LLM (for example a Google AI Studio key).

## Installation

### 1. Clone the repository and enter the project directory

```powershell
git clone <repository-url>
cd norag-rag
```

If the repository is already on your computer, open PowerShell in its root directory instead.

### 2. Create and activate a Python virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the LLM

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` (or create `.env.local`, which wins over `.env`) and configure:

```text
LLM_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your_api_key
LLM_MODEL=gemini-3.5-flash-lite
```

`LLM_URL` can point to any OpenAI-compatible chat completions API. An old `.env` with only `GOOGLE_API_KEY` still works (Gemini is the fallback). Restart the backend after changing it.

Use your own key. Never commit `.env` or expose the key in evaluation files or screenshots.

### 5. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## Running the application

Start the backend from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

The FastAPI service is available at:

```text
http://127.0.0.1:8000
```

The interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

In a second PowerShell terminal, start the frontend:

```powershell
cd frontend
npm run dev
```

Open the local URL printed by Vite in the terminal.

## Using the application

1. Select **Vellora** or **Kestrel** using the corpus selector.
2. View the documents belonging to the selected corpus.
3. Upload, index or delete documents as needed.
4. Rebuild the selected corpus index when its source documents change.
5. Select the relevant documents and submit a question.
6. Inspect the generated answer and its source references.
7. Open the **Words** tab to inspect corpus vocabulary or analyze the terms in a question.

Each corpus has its own document directory and persistent ChromaDB collection. Rebuilding one corpus should not delete the other corpus's index.

## Running the evaluation

Run commands from the project root with the Python virtual environment activated.

### Retrieval evaluation

Run the evaluation runner:

```powershell
python .\evaluation\evaluate.py
```

To evaluate a particular corpus and configuration:

```powershell
python .\evaluation\evaluate.py --corpus vellora --config hybrid
python .\evaluation\evaluate.py --corpus vellora --config semantic

python .\evaluation\evaluate.py --corpus kestrel --config hybrid
python .\evaluation\evaluate.py --corpus kestrel --config semantic
```

The configurations are:

- **Hybrid:** exact error-code matching enabled.
- **Semantic:** exact error-code matching disabled.

The evaluation uses fixed question sets and reports HitRate@1, @3, @5 and @10. Unanswerable questions are excluded from retrieval HitRate calculations.

Results are saved as JSONL files in `evaluation/`, with aggregate results in:

```text
evaluation/summary_v2.csv
```

### No-context evaluation

The no-context script sends questions to Gemini without retrieved documents:

```powershell
python .\evaluation\no_context.py
```

The generated answers are stored separately for Vellora and Kestrel:

```text
evaluation/no_context_vellora.jsonl
evaluation/no_context_kestrel.jsonl
```

The manually assessed results are stored in:

```text
evaluation/no_context_grading_v2.csv
```

### RAG answer generation

To generate answers from saved hybrid retrieval results:

```powershell
python .\evaluation\evaluate.py --corpus vellora --generate-answers
python .\evaluation\evaluate.py --corpus kestrel --generate-answers
```

Generated answers are stored in:

```text
evaluation/answers_v2_vellora_hybrid.jsonl
evaluation/answers_v2_kestrel_hybrid.jsonl
```

These commands call Gemini and may consume API quota. Existing completed answer files should be preserved when repeating evaluation work.

### Hand-grading

The generated RAG answers are manually graded in:

```text
evaluation/grading_v2.csv
```

The no-context answers are manually graded in:

```text
evaluation/no_context_grading_v2.csv
```

The RAG grading categories are:

- `correct`
- `partial`
- `wrong`
- `unsupported`

The no-context correctness column records whether the model answered each question correctly without retrieval.

## Evaluation artifacts

The evaluation directory contains:

| File | Purpose |
|---|---|
| `questions.vellora.jsonl` | Vellora evaluation questions |
| `questions.kestrel.jsonl` | Kestrel evaluation questions |
| `no_context_vellora.jsonl` | Gemini answers without retrieval for Vellora |
| `no_context_kestrel.jsonl` | Gemini answers without retrieval for Kestrel |
| `no_context_grading_v2.csv` | Human assessment of no-context answers |
| `results_v2_vellora_hybrid.jsonl` | Vellora hybrid retrieval results |
| `results_v2_vellora_semantic.jsonl` | Vellora semantic retrieval results |
| `results_v2_kestrel_hybrid.jsonl` | Kestrel hybrid retrieval results |
| `results_v2_kestrel_semantic.jsonl` | Kestrel semantic retrieval results |
| `summary_v2.csv` | Aggregate and per-question-type retrieval metrics |
| `answers_v2_vellora_hybrid.jsonl` | Generated Vellora RAG answers |
| `answers_v2_kestrel_hybrid.jsonl` | Generated Kestrel RAG answers |
| `grading_v2.csv` | Human assessment of generated RAG answers |

## V4: observability, guardrails, graphs for any document

### Tabs

- **Ask documents:** every answer has a **View trace** link. **Conversation mode** keeps the previous questions and answers; a follow-up question is first rewritten into a standalone question (one extra LLM call), and the UI shows what was searched for.
- **Concept graph:** build the concept graph of the selected documents (or of the whole corpus) of any corpus, including uploaded documents, then explore it: search entities, filter by type, click through neighbours, read the quote behind each relation, and ask about an entity. Kestrel keeps its V3 schema; other documents use a general schema. Once built, the graph can be switched on as a retriever.
- **Traces:** one trace per question, upload, re-index and graph build, with its spans (guard, rewrite, each retriever, fusion, generation, validation), timings, tokens, cost and configuration fingerprint.
- **Metrics:** computed from the traces: outcomes (answered, abstained, invalid, error, blocked), p50/p95 per request and per step, tokens and cost per question, per indexed document and per day, and a warning when the running configuration was not evaluated.

### Guardrails

- **Injection scanner** (`backend/scanner.py`): rules with a name and a severity. An uploaded document with a high-severity finding is not indexed; it is moved to `quarantine/<corpus>/` with a JSON report. At question time, retrieved chunks with a finding are withheld from the prompt, and a question that tries to change the instructions is refused (outcome `blocked`). Flagged chunks are also skipped by graph extraction.
- **Personal data redaction** (`backend/redact.py`): emails, phone numbers, IBANs, card numbers and IP addresses are replaced in traces. Emails, phones, IBANs and card numbers are also replaced in what is sent to the LLM (`REDACT_LLM_INPUT`). Role mailboxes such as `billing@` are kept.
- **Graph budget:** a graph build that needs more than `GRAPH_MAX_NEW_CHUNKS` new LLM calls is refused before any call.

### Graph from the command line

```powershell
python -m backend.graph_build --corpus vellora --document vellora-duo-espresso-machine.txt
```

### Tests

The tests use a fake LLM and a fake embedding model: no key, no network, no cost.

```powershell
python -m pytest -q tests
```

### Final evaluation re-run

```powershell
python .\evaluation\evaluate.py --release v4 --config dense dense+bm25 dense+bm25+graph
python .\evaluation\compare_releases.py
```

The first command writes `evaluation/results_v4_*.jsonl` and `evaluation/summary_v4.csv` (with the configuration fingerprint); the second puts the overall rows of V2, V3 and V4 side by side in `evaluation/comparison_v4.csv`. The graph configurations run on every corpus that has a graph.

## Important notes

- The `.env` file must remain private and must not be committed.
- The ChromaDB collections are persistent and corpus-specific.
- Rebuilding an index can take time because embeddings are computed locally.
- Evaluation answer generation uses the configured LLM and is subject to API rate limits.
- `traces/` and `quarantine/` are local and ignored by git. Traces older than `TRACE_RETENTION_DAYS` are deleted.
- HitRate measures retrieval of expected evidence; it does not by itself measure answer correctness.
- The original standalone `ask_norag_google.py` and `ask_rag_google.py` scripts are retained as earlier project experiments. The multi-corpus web application is the main V2 application.

## Version control

The V2 release is tagged locally as `v2` after final verification. Push changes to the remote repository only after agreement from the team.