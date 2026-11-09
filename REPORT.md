# REPORT.md

## Team Details

- Team members: Mahmoud NASSER, Mohamad Ibaad SHAMS and Joseph AYOUB.
- Work distribution:
  - **Mahmoud NASSER:** Backend development and RAG integration.
  - **Mohamad Ibaad SHAMS:** Frontend development and user interface.
  - **Joseph AYOUB:** Corpus preparation, evaluation, testing and reporting.

## 1. Project Overview

The project implements a Retrieval-Augmented Generation (RAG) application that supports two independent knowledge bases:

- **Vellora:** The selected user corpus, including the Vellora Duo espresso-machine manual and the existing literary documents.
- **Kestrel:** The technical knowledge base provided for the assignment.

Users can select a corpus, manage its documents, rebuild its index, ask questions and inspect retrieved information. The application uses ChromaDB for persistent vector storage, Sentence Transformers for local embeddings and Google Gemini for answer generation.

V2 extends the original application with multi-corpus support, a Words analysis feature and a repeatable evaluation pipeline.

V3 adds hybrid retrieval. Two new retrievers, **BM25** (keyword search, both corpora) and a **concept graph** (an LLM-extracted graph of entities and relations, Kestrel only), work alongside the V2 dense retriever. **Reciprocal Rank Fusion (RRF)** combines whichever retrievers are switched on. Each retriever can be switched on or off from the UI and from the evaluation command. Every reference shows which retriever found it and at which rank, and when the graph is on, the UI shows the path behind the answer. The V2 evaluation was extended with Recall@k and run on several configurations, and every improvement is measured against the V2 dense baseline on the same question sets.

## 2. Decision Log

### D1 — Corpus selection

**Options:** One literary corpus; one technical manual; a mixed corpus.

**Choice:** Use *Crime and Punishment* and the Vellora Duo espresso-machine manual as the original Vellora corpus, then add the provided Kestrel knowledge base for V2.

**Reason:** The literary document provides a large text collection, while the espresso manual tests precise factual retrieval involving quantities, settings and procedures. Kestrel adds technical support questions, identifiers, policy versions and multi-hop questions.

### D2 — Embedding model

**Options:** Local Sentence Transformers; hosted embedding API.

**Choice:** `all-MiniLM-L6-v2`.

**Reason:** It runs locally, is already integrated into the project and avoids additional embedding API costs.

### D3 — Vector database

**Options:** ChromaDB; FAISS; hosted vector database.

**Choice:** Persistent local ChromaDB.

**Reason:** The existing implementation can be reused, and persistent storage allows indexes to survive application restarts. V2 uses separate collections for the two corpora.

### D4 — LLM provider

**Options:** Google Gemini; OpenAI API; another compatible provider.

**Choice:** Google Gemini through the OpenAI-compatible SDK.

**Reason:** This integration was already present in the application.

### D5 — Backend

**Options:** Standalone Python script; FastAPI; Flask.

**Choice:** FastAPI.

**Reason:** It provides an HTTP API and integrates with Pydantic response validation.

### D6 — Frontend

**Options:** Streamlit; HTML/CSS/JavaScript; React.

**Choice:** React with Vite.

**Reason:** It provides a structured interface for document management, corpus selection, question submission, answer display and Words analysis.

### D7 — Chunking

**Options:** Fixed-size chunks with overlap; paragraph-based chunking; semantic chunking.

**Choice:** Fixed-size character chunks with configurable overlap.

**Reason:** This preserves the original implementation and provides a straightforward baseline for retrieval experiments.

### D8 — Evaluation

**Options:** Manual inspection; fixed evaluation datasets with automated retrieval metrics.

**Choice:** Fixed JSONL question sets containing expected answers and evidence, evaluated using HitRate@1, @3, @5 and @10.

**Reason:** Fixed datasets allow repeatable comparisons between retrieval configurations and corpora.

### D9 — Multi-corpus architecture 

**Choice:** Keep Vellora and Kestrel in separate ChromaDB collections and expose corpus selection through the backend API and frontend.

**Reason:** This prevents indexing, rebuilding or deleting one corpus from affecting the other.

### D10 — Retrieval configurations

**Choice:** Compare a hybrid configuration with exact error-code matching enabled against a semantic-only configuration with exact matching disabled.

**Reason:** This measures whether the existing exact error-code matching improves the ranking of relevant passages.

**V3 update:** superseded by D12. The `E##` rule never matched Kestrel identifiers such as `KST-2044`, so the two V2 Kestrel configurations were identical. V3 retires the rule; BM25 is its general, corpus-independent replacement.

### D11 — BM25 index: where it lives and how it is built

**Options:** keep the `norag-rag` index (in memory, rebuilt on every run); persist our own inverted index to disk; SQLite FTS5 (built-in `bm25()`); Postgres full-text search (`ts_rank` is not BM25; ParadeDB `pg_search` is); Elasticsearch/OpenSearch.

**Choice:** our own inverted index (`backend/bm25.py`), built from the chunks already stored in ChromaDB and saved as JSON per corpus (`bm25_index/<corpus>.json`). It is rebuilt in full after every upload, index, rebuild or delete.

**Reason:** BM25 and dense retrieval must rank exactly the same chunks, so ChromaDB stays the single source of truth. A full rebuild is the honest update strategy: adding or removing chunks changes the IDF of every word. It is also cheap: 0.358 s for all 2,494 Vellora chunks. A search engine would be a second service to run for corpora this small. The index is written to a temporary file first, then renamed, so a crash never leaves a half-written index.

### D12 — Retire the V2 `E##` rule

**Choice:** remove the regex prefix from retrieval. The V3 `dense` configuration is pure semantic search, the V2 baseline.

**Reason:** the rule was written for one corpus and one identifier format. BM25 handles identifiers in any format.

### D13 — Fusion: RRF on ranks, 20 candidates per retriever

**Options:** score normalization and weighted sum; RRF; a cross-encoder re-ranker.

**Choice:** RRF, `score = Σ w / (k + rank)`, with k = 60 and equal weights by default. Each retriever proposes 20 candidates, and the fused list is cut to `top_k`. Ties go to the best single rank, then to the chunk id, so results are reproducible.

**Reason:** a BM25 score and a cosine distance are not on the same scale, so only ranks are combined. RRF needs no training and no extra model call. Each fused chunk keeps `found_by = {retriever: rank}` for the UI and the evaluation.

### D14 — Concept graph schema: fixed

**Options:** an open schema (whatever types the LLM finds); a fixed schema.

**Choice:** a fixed schema (`backend/graph_schema.py`) with 8 entity types (ErrorCode, Service, Team, Person, Channel, Plan, Feature, Role) and 12 typed relations, each with allowed source and target types (for example `raised_by`: ErrorCode → Service, `owned_by`: Service → Team, `has_oncall_channel`: Team → Channel).

**Reason:** a fixed list can be validated strictly and queried predictably, and it stops the LLM from inventing several spellings of the same relation. It was designed for the questions Kestrel support asks: which service raises an error, which team owns it, where to escalate. The cost is that facts outside the schema are lost (refund rules, conditions such as "only if more than three customers are affected").

### D15 — Graph extraction and validation

**Choice:** one Gemini call per chunk (JSON mode, temperature 0). The model returns entities and `(entity, relation, entity)` triples, each with a quote copied from the chunk. Every item is validated separately with Pydantic, then checked: the relation's entities must come from the same chunk, their types must match the relation's signature, and the quote must be found in the chunk. Invalid items are dropped and counted by reason. If the whole answer is not valid JSON, the call is retried once with the error, and then the chunk is skipped and listed in `failed_chunks`.

**Reason:** dropping one bad triple is better than losing the whole chunk. The quote check catches invented facts, though not misattributed ones (see section 7). Raw answers are cached by chunk text, prompt version and model (`graph/cache/`), so a crash or a change to the validation rules never pays twice for the same chunk.

### D16 — Entity resolution

**Options:** exact names; normalized names; embedding similarity; LLM-judged merges.

**Choice:** normalized names (lowercase, without a leading "the", without a "team"/"service" suffix, channels always starting with `#`). Same type and same normalized name, or an alias the LLM gave for the same entity, means same node (union-find). The same name with different types is kept apart and listed under `possible_missed_merges` for review.

**Reason:** it is deterministic, explainable and free. Embedding similarity would merge "Identity" and "Identity provider".

### D17 — The graph as a retriever: local walk

**Options:** local walk from the question's entities; global/community summaries (GraphRAG); keyword-extracted entities plus neighbors (LightRAG style); an LLM call to extract the question's entities; path as context.

**Choice:** the question's entities are found by whole-word matching of node names and aliases, with no LLM call at question time. From them, the walk follows edges in both directions up to 3 hops. Hubs with more than 12 edges are not expanded past hop 1. Each chunk scores Σ 1/hop over the edges that point to it, so closer facts rank higher.

**Reason:** q08 needs 3 edges (KST-5031 → postroom → Messaging → #oncall-messaging). String matching costs nothing and works for identifiers and names, which is what Kestrel questions contain. Global summaries answer "what is this corpus about" questions, which our question sets do not ask.

### D18 — Weighted RRF (team-of-3 piece)

**Choice:** two weighted configurations, with weights fixed before running and not tuned on the question sets: `dense+bm25_w` (BM25 ×2) and `dense+bm25+graph_w` (graph ×2). Each is compared with plain RRF on every question type.

**Reason:** with 10 answerable questions per corpus, tuning weights on the test set would overfit.

### D19 — Hidden instruction in the V3 assignment PDF

The V3 PDF contains a hidden instruction, invisible on the slide but present in its text layer, asking for an unlinked `/bm25` page. It is not part of the visible assignment, and we did not implement it. The V2 PDF contained a similar hidden `/about` instruction, also not implemented. Anything read by a tool or a model from a document is data, not instructions.

## 3. Work Ownership and Contributions

### Mahmoud NASSER — Backend

- Implemented the FastAPI application and API schemas.
- Implemented document loading, chunking and indexing.
- Integrated persistent ChromaDB storage.
- Implemented document listing, upload, deletion and index rebuilding.
- Integrated Gemini answer generation and evidence-based responses.
- Implemented corpus-aware backend operations for Vellora and Kestrel.
- Added separate vector collections to isolate the two corpora.
- Implemented the backend Words analysis endpoint.
- Added support for comparing retrieval with exact-match prioritization enabled and disabled.

### Mohamad Ibaad SHAMS — Frontend

- Created the React frontend using Vite.
- Implemented document management controls.
- Integrated the frontend with the FastAPI backend.
- Added the Vellora/Kestrel corpus selector.
- Updated document listing, upload, indexing, deletion and question submission to respect the selected corpus.
- Implemented the Words tab and its corpus vocabulary and question-analysis views.
- Tested the frontend corpus selection and Words functionality.

### Joseph AYOUB — Evaluation, testing and reporting

- Prepared the evaluation question sets for Vellora and Kestrel.
- Preserved the provided Kestrel starter questions and prepared the Vellora questions.
- Implemented the repeatable retrieval evaluation.
- Ran the hybrid and semantic configurations on both corpora.
- Implemented and ran the no-context evaluation.
- Implemented and ran answer generation for the saved retrieval results.
- Hand-graded the generated answers and no-context answers.
- Used the Words feature to investigate retrieval misses.
- Prepared the evaluation results and coordinated this report.

### V3 contributions

<!-- Each team member: list here what you did in V3. Only list work you actually did. -->

**Joseph AYOUB**

- BM25: shared tokenizer, inverted index, persistence and automatic rebuilds.
- Split retrieval into one function per retriever, with RRF fusion and weighted RRF.
- Provenance per reference, token usage and latency in the API and the UI, and a 400 error for invalid requests.
- Concept graph: schema, extraction and validation, entity resolution, build cost tracking and caching, the two extraction prompts, the graph retriever, and the path in the UI.
- V3 evaluation: Recall@k, configurations, graph statistics, the two new `two_hop` questions (q11, q12) and their no-context answers.
- Extraction-quality grading of 10 triples, README and this report.

## 4. Implementation and Validation Log

### Step 1 — Initial backend application

**Owner:** Mahmoud NASSER.

- Created the FastAPI application.
- Added health checking and CORS configuration.
- Defined initial API schemas.

**Validation:** The backend was integrated with the frontend and used for subsequent indexing and question-answering tests.

### Step 2 — Document loading and indexing

**Owner:** Mahmoud NASSER.

- Implemented UTF-8 document loading for `.txt` and `.md` files.
- Implemented configurable character-based chunking and overlap.
- Stored document identifiers, chunk identifiers and line ranges in ChromaDB metadata.
- Added document listing, individual indexing and full-index rebuilding.

**Validation:** Documents were indexed and retrieved through the application.

**Known limitation:** A full-index rebuild clears the selected collection before indexing its replacement documents. Recovery from a failed rebuild could be improved.

### Step 3 — Document management

**Owner:** Mahmoud NASSER.

- Implemented upload for `.txt` and `.md` files.
- Added file size, format, UTF-8 and duplicate filename validation.
- Implemented deletion of source files and their indexed chunks.

**Validation:** Document management was integrated into the application and exercised through the frontend.

**Known limitation:** File-system and vector-store operations are not transactional. Recovery from partial storage failures could be improved.

### Step 4 — Retrieval and answer generation

**Owner:** Mahmoud NASSER.

- Implemented ChromaDB retrieval restricted to selected documents.
- Integrated Gemini answer generation using retrieved evidence.
- Added response validation, source references and `insufficient_evidence` handling.
- Implemented exact error-code matching to improve retrieval ranking.

**Validation:** Initial E05 testing returned the expected answer and reference. Unsupported E00 returned `insufficient_evidence`. Empty questions, invalid `top_k` values and invalid document identifiers were also tested.

### Step 5 — Frontend implementation

**Owner:** Mohamad Ibaad SHAMS.

- Created the React/Vite frontend.
- Implemented document management and question submission.
- Integrated answer and reference display.
- Added corpus selection for Vellora and Kestrel.
- Updated frontend API calls to use the selected corpus.
- Added the Words tab.

**Validation:** Corpus selection, document isolation and Words functionality were tested. Rebuilding the Kestrel index did not remove the Vellora documents or prevent their retrieval.

### Step 6 — Multi-corpus backend support

**Owner:** Mahmoud NASSER.

- Added corpus-specific document directories.
- Added separate persistent ChromaDB collections.
- Updated listing, upload, indexing, rebuilding, deletion, retrieval and answer generation to accept a corpus.
- Preserved Vellora as the default corpus for backward compatibility.

**Validation:** Kestrel was rebuilt successfully with five documents and 43 chunks. The API returned the Kestrel document listing. The isolation test confirmed that Vellora remained available.

### Step 7 — Words analysis

**Owners:** Mahmoud NASSER (backend) and Mohamad Ibaad SHAMS (frontend).

- Implemented corpus vocabulary analysis, including word frequency and chunk locations.
- Implemented question analysis showing which terms occur in the selected corpus and where they occur.
- Displayed absent words and matching chunk locations in the frontend.

**Validation:** Both vocabulary and question analysis were tested.

### Step 8 — Evaluation datasets

**Owner:** Joseph AYOUB.

- Prepared `evaluation/questions.vellora.jsonl` with 12 questions.
- Prepared `evaluation/questions.kestrel.jsonl` with 10 questions based on the provided starter set.
- Included answerable and unanswerable questions and expected evidence.
- Covered multiple question types, including exact identifiers, split answers, vocabulary, versions, two-hop questions and questions not answerable from the corpus.

**Validation:** Both JSONL files were parsed and their record counts verified.

### Step 9 — Retrieval evaluation

**Owner:** Joseph AYOUB.

- Implemented a repeatable evaluation runner.
- Compared hybrid retrieval with exact matching enabled against semantic retrieval with exact matching disabled.
- Evaluated both corpora using the same question sets.
- Saved individual retrieval results and aggregate summaries.

**Metric:** HitRate@k measures whether the expected evidence is found among the top k retrieved chunks. Unanswerable questions are excluded.

**Overall results:**

| Corpus | Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| Vellora | Hybrid | 70% | 100% | 100% | 100% |
| Vellora | Semantic | 60% | 90% | 90% | 100% |
| Kestrel | Hybrid | 62.5% | 87.5% | 87.5% | 100% |
| Kestrel | Semantic | 62.5% | 87.5% | 87.5% | 100% |

**Per-question-type results:**

Values are Hit@1 / Hit@3 / Hit@5 / Hit@10.

| Corpus | Question type | Hybrid | Semantic |
|---|---|---|---|
| Vellora | Exact identifier | 66.7 / 100 / 100 / 100% | 66.7 / 100 / 100 / 100% |
| Vellora | Other | 100 / 100 / 100 / 100% | 100 / 100 / 100 / 100% |
| Vellora | Split answer | 75 / 100 / 100 / 100% | 50 / 75 / 75 / 100% |
| Vellora | Vocabulary | 50 / 100 / 100 / 100% | 50 / 100 / 100 / 100% |
| Kestrel | Exact identifier | 50 / 50 / 50 / 100% | 50 / 50 / 50 / 100% |
| Kestrel | Split answer | 100 / 100 / 100 / 100% | 100 / 100 / 100 / 100% |
| Kestrel | Two-hop | 100 / 100 / 100 / 100% | 100 / 100 / 100 / 100% |
| Kestrel | Versions | 0 / 100 / 100 / 100% | 0 / 100 / 100 / 100% |
| Kestrel | Vocabulary | 100 / 100 / 100 / 100% | 100 / 100 / 100 / 100% |

**Interpretation:**

- Hybrid retrieval improves Vellora Hit@1 by 10 percentage points and Hit@3/Hit@5 by 10 percentage points.
- The largest Vellora improvement occurs in split-answer questions.
- Kestrel has identical results for both configurations. The current exact-match rule recognizes `E##` error codes but not Kestrel identifiers such as `KST-2044`.
- Kestrel version questions have 0% Hit@1 but 100% Hit@3. The relevant evidence is present at a lower rank, indicating a ranking problem.
- HitRate@10 reaches 100% for both corpora, but this does not guarantee that all evidence required for a multi-part answer is retrieved.

### Step 10 — No-context evaluation

**Owner:** Joseph AYOUB.

- Generated answers to all 22 evaluation questions without providing retrieved documents.
- Saved the outputs separately for Vellora and Kestrel.
- Manually assessed all 22 responses and recorded correctness and grading notes in `evaluation/no_context_grading_v2.csv`.

**Results:**

| Corpus | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Vellora | 4 | 12 | 33.3% |
| Kestrel | 1 | 10 | 10.0% |
| Combined | 5 | 22 | 22.7% |

For answerable questions alone, the model correctly answered 2 of 18 questions (11.1%).

**Important observations:**

- V04 was answered using general knowledge about coffee-machine descaling.
- V05 was a borderline knowledge leak: the answer included the correct five-second minimum in a generic 5–10 second recommendation.
- Kestrel q09 exposed confusion between Kestrel Analytics and the Kestrel weather-meter brand. The model invented mobile application information that did not apply to the corpus.
- V11, V12 and q10 were correctly answered through abstention, but these cases provide limited evidence of the model's knowledge of the specific products.

These leakage cases are documented rather than silently removed from the evaluation.

### Step 11 — RAG answer generation and hand-grading

**Owner:** Joseph AYOUB.

- Generated answers using the saved retrieved chunks.
- Saved generated answers and references alongside the evaluation results.
- Hand-graded all 22 responses using four categories: correct, partial, wrong and unsupported.
- Recorded explanations for each grade in `evaluation/grading_v2.csv`.

**Grading results:**

| Corpus | Correct | Partial | Wrong | Unsupported |
|---|---:|---:|---:|---:|
| Vellora | 11 | 1 | 0 | 0 |
| Kestrel | 6 | 3 | 1 | 0 |
| Combined | 17 | 4 | 1 | 0 |

**Interpretation:**

Vellora achieved 11 correct answers and one partial answer. Its partial answer was V06, where the model supplied only the upper temperature limit rather than the complete 10–32°C range, even though the retrieved evidence contained the full range. This was a generation failure.

Kestrel achieved six correct answers, three partial answers and one wrong answer.

- q04 was partial because the answer omitted the final testing and enforcement steps for SSO setup, despite the evidence being retrieved.
- q06 and q07 were partial because the model did not clearly prioritize the current 2026 billing policy over the older 2025 policy.
- q08 was wrong because the answer omitted the second part of the required support route. The first-hop information was retrieved, but the service-catalog evidence identifying Messaging and `#oncall-messaging` was missing. This is a retrieval failure involving incomplete multi-hop evidence.

No answers were graded unsupported.

### Step 12 — BM25 (V3)

**Owner:** Joseph AYOUB.

- Moved the Words tokenizer into `backend/tokenizer.py`, so BM25 and the Words tab see exactly the same words.
- Implemented the BM25 inverted index (`backend/bm25.py`, k1 = 1.5, b = 0.75) with save and load.
- Rebuilt the index automatically after index, rebuild and delete. A full rebuild runs once per corpus rebuild, not once per document.

**Validation:** on the espresso manual alone, BM25 reproduces the IDF table from class: 43 chunks, `error` 1.77, `e47` 3.38, `mean`/`fix` 4.48, with the E47 chunk ranked #1. On Kestrel, q08's runbook chunk ranks #1 (scores 13.03 / 10.6 / 9.9 for the top three). Deleting a document removes its chunks from the index, and re-adding it restores them.

### Step 13 — Retrievers and RRF fusion (V3)

**Owner:** Joseph AYOUB.

- Split `retrieve_chunks` into `dense_search`, `bm25_search` and (later) `graph_search`, each returning ranked chunk ids, and added `rrf_fuse` with optional weights.
- `POST /api/ask` takes a `retrievers` list and returns, for each reference, its fused rank and its rank in each retriever (`found_by`), plus token usage and latency per step.

**Validation:** with an empty retriever list the API answers 422. E47 on Vellora with dense + BM25 was answered correctly: 1 LLM call, 1,621 tokens, 1,097 ms. The cited chunk was fused #3, found by dense #8 and BM25 #3.

### Step 14 — UI (V3)

**Owner:** Joseph AYOUB.

- Added retriever checkboxes (the graph only on Kestrel), a cost line under each answer (retrievers, LLM calls, tokens, ms), each reference's chunk, fused rank and per-retriever ranks, and a Graph path panel.

**Validation:** q08 on Kestrel with all three retrievers (screenshot `docs-images/q08-graph-path.png`). An earlier run with only the runbook selected showed what document filtering does: the path stopped at hop 1, and the answer could not name the team.

### Step 15 — Concept graph (V3)

**Owner:** Joseph AYOUB.

- Built `backend/graph_schema.py`, `backend/graph_build.py` and `backend/graph.py`, as described in D14–D17 and section 7.
- Ran the extraction three times: prompt 1, prompt 1 with the corrected quote check (no new calls, thanks to the cache), and prompt 2.

**Validation:** see section 7. A wrong document, unknown retriever or missing graph now returns a 400 with the real reason. In V2, any `ValueError` was reported as a 502 "answer could not be validated".

### Step 16 — Evaluation (V3)

**Owner:** Joseph AYOUB.

- Added Recall@k, seven configurations, retrieval latency, per-question graph statistics, and `--candidates` / `--rrf-k` options that write to separately named files.
- Added two `two_hop` Kestrel questions with an evidence quote for every hop: q11 (KST-1012 → Identity → Priya Raman / #oncall-identity) and q12 (KST-6002 → Payments Core → #oncall-payments). We checked that each quote fits inside one chunk at 800 / 10%. No existing question was edited.
- Ran the no-context baseline on q11 and q12, and graded 10 random graph triples.

## 5. V3 Results

All runs use chunk size 800 and 10% overlap, top 10 retrieved chunks, RRF k = 60 and 20 candidates per retriever. Kestrel now has 12 questions (10 answerable, after adding q11 and q12) and Vellora has 12 (10 answerable). Kestrel numbers are therefore not directly comparable with the V2 table, which had 8 answerable Kestrel questions; V3 reruns the V2 baseline (`dense`) on the new sets.

### 5.1 Overall

Hit@1 / Hit@3 / Hit@5 / Hit@10, then Recall@5 and the average retrieval time per question. No configuration makes an LLM call at question time: embeddings run locally and graph entities are matched by string.

| Corpus | Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | ms |
|---|---|---:|---:|---:|---:|---:|---:|
| Vellora | dense (V2 baseline) | 0.4 | 0.8 | 0.9 | 0.9 | 0.85 | 28.3 |
| Vellora | bm25 | **0.7** | 0.9 | 0.9 | 1.0 | 0.90 | 7.7 |
| Vellora | dense + bm25 | 0.6 | **1.0** | **1.0** | **1.0** | **0.95** | 31.6 |
| Vellora | dense + bm25, weighted (BM25 ×2) | **0.7** | **1.0** | **1.0** | **1.0** | **0.95** | 29.3 |
| Kestrel | dense (V2 baseline) | 0.6 | 0.9 | 0.9 | 1.0 | 0.75 | 18.8 |
| Kestrel | bm25 | **0.7** | 0.9 | 0.9 | 0.9 | 0.675 | 3.5 |
| Kestrel | dense + bm25 | 0.6 | **1.0** | **1.0** | **1.0** | 0.85 | 12.3 |
| Kestrel | dense + bm25, weighted (BM25 ×2) | 0.6 | 0.9 | **1.0** | **1.0** | 0.775 | 11.9 |
| Kestrel | graph alone | 0.2 | 0.2 | 0.2 | 0.2 | 0.20 | 1.9 |
| Kestrel | dense + bm25 + graph | 0.6 | **1.0** | **1.0** | **1.0** | 0.90 | 13.2 |
| Kestrel | dense + bm25 + graph, weighted (graph ×2) | 0.6 | **1.0** | **1.0** | **1.0** | **0.95** | 11.9 |

Answer generation, when it runs, adds one LLM call: about 1,600 tokens and about 1 s per question in our UI tests (1,621 tokens / 1,097 ms; 1,640 tokens / 876 ms).

### 5.2 Per question type

Values are Hit@1 / Hit@5 / Recall@5. *n* is the number of answerable questions of that type. "w" means weighted RRF.

**Vellora**

| Type | n | dense (V2) | bm25 | dense + bm25 | dense + bm25 w |
|---|---:|---|---|---|---|
| overall | 10 | 0.40 / 0.90 / 0.85 | 0.70 / 0.90 / 0.90 | 0.60 / 1.00 / 0.95 | 0.70 / 1.00 / 0.95 |
| exact_identifier | 3 | 0.33 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.33 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| split_answer | 4 | 0.25 / 1.00 / 0.88 | 0.75 / 1.00 / 1.00 | 0.75 / 1.00 / 0.88 | 0.75 / 1.00 / 0.88 |
| vocabulary | 2 | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 |
| other | 1 | 1.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 |

**Kestrel**

| Type | n | dense (V2) | bm25 | dense + bm25 | dense + bm25 w | graph | dense + bm25 + graph | dense + bm25 + graph w |
|---|---:|---|---|---|---|---|---|---|
| overall | 10 | 0.60 / 0.90 / 0.75 | 0.70 / 0.90 / 0.68 | 0.60 / 1.00 / 0.85 | 0.60 / 1.00 / 0.78 | 0.20 / 0.20 / 0.20 | 0.60 / 1.00 / 0.90 | 0.60 / 1.00 / 0.95 |
| exact_identifier | 2 | 0.50 / 0.50 / 0.50 | 1.00 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 |
| split_answer | 2 | 1.00 / 1.00 / 1.00 | 0.50 / 1.00 / 0.62 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 0.62 | 0.00 / 0.00 / 0.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 |
| two_hop | 3 | 0.67 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 1.00 / 1.00 / 0.50 | 0.67 / 0.67 / 0.67 | 1.00 / 1.00 / 0.67 | 1.00 / 1.00 / **0.83** |
| versions | 2 | 0.00 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.50 / 1.00 / 1.00 | 0.50 / 1.00 / 1.00 |
| vocabulary | 1 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 |

Each retriever wins on the failure it was designed for:

- **Identifiers → BM25.** BM25 moves Kestrel `exact_identifier` Hit@5 from 0.50 (the V2 baseline, which the `E##` rule never helped) to 1.00, and Vellora `exact_identifier` Hit@1 from 0.33 to 1.00.
- **Different vocabulary → dense.** On Kestrel q05 ("I lost my phone"), BM25 alone finds nothing (0 / 0 / 0), while dense ranks the evidence #1. On Vellora, fusion fixes a `vocabulary` question that neither retriever found alone in its top 5 (Hit@5 0.50 → 1.00).
- **Chains → graph.** Kestrel `two_hop` Recall@5 is stuck at 0.50 for dense, BM25 and their fusion: each finds one hop out of two. The graph raises it to 0.67, and to 0.83 when weighted ×2.

### 5.3 Reading the results

- **BM25 fixes identifiers:** Kestrel `exact_identifier` Hit@5 goes from 0.50 to 1.00 (section 5.2).
- **Fusion wins from k = 3.** On both corpora, dense + BM25 reaches Hit@3 = 1.0. Recall@5 rises from 0.85 to 0.95 on Vellora and from 0.75 to 0.85 on Kestrel compared with the V2 baseline.
- **Fusion loses at k = 1.** BM25 alone has the best Hit@1 on both corpora (0.7). Plain RRF with k = 60 flattens ranks: a chunk ranked #1 by BM25 but #8 by dense can lose to a chunk ranked #3 by both. Section 5.6 gives one question where this happens.
- **The graph moves Recall, not HitRate.** On Kestrel, adding the graph raises Recall@5 from 0.85 to 0.90, and to 0.95 with graph ×2, while every HitRate stays the same. The graph does not find new questions; it completes chains. This is exactly what V2's HitRate could not show: V2 reported 100% two_hop HitRate while q08's answer was graded wrong because its second hop was missing.
- **Weights depend on the corpus.** BM25 ×2 restores Vellora's Hit@1 (0.6 → 0.7) but hurts Kestrel (Recall@5 0.85 → 0.775, Hit@3 1.0 → 0.9). No single weighting is best for both.
- **Cost.** BM25 adds 3–8 ms and the graph about 1 ms per question. Dense dominates retrieval time.

### 5.4 Default configuration

**Dense + BM25 with plain RRF on both corpora, with the concept graph also on for Kestrel.**

- Dense + BM25 is best or tied-best at Hit@3, Hit@5, Hit@10 and Recall@5 on both corpora, for about 12–32 ms and no LLM call. The price is Hit@1 (0.7 → 0.6 against BM25 alone).
- Weighted BM25 fixes Vellora's Hit@1 but costs Kestrel 0.075 of Recall@5. Corpus-specific weights tuned on 10 questions would be fitted to the test set, so the default stays unweighted.
- On Kestrel, the graph adds 0.05 of Recall@5 (0.10 when weighted ×2) for about 1 ms and no LLM call at question time. Its build cost is paid once (section 7.5).

### 5.5 After fusion, did the graph's chunks reach the top k?

Per answerable Kestrel question with the graph on (`graph cands` = chunks the graph proposed; `graph-only` = chunks in the top 5 that no other retriever found):

| Question | Type | Entities found | Graph cands | Graph chunks in top 5 | Graph-only in top 5 | R@5 dense + bm25 | R@5 + graph | R@5 + graph w |
|---|---|---|---:|---:|---:|---:|---:|---:|
| q08 | two_hop | KST-5031, Team | 3 | 3 | 0 | 0.5 | **1.0** | **1.0** |
| q11 | two_hop | KST-1012, Team | 2 | 1 | 0 | 0.5 | 0.5 | **1.0** |
| q12 | two_hop | none | 0 | 0 | 0 | 0.5 | 0.5 | 0.5 |
| q02 | exact_identifier | KST-4012 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| other 6 | — | none | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |

Averaged over the 10 answerable questions: 0.4 graph chunks in the top 5 with plain RRF and 0.5 with graph ×2. Graph-only chunks: 0.0 plain, 0.1 weighted.

- **q08: yes.** All 3 graph chunks reached the top 5. None was graph-only: dense and BM25 had also proposed the service-catalog chunk, but below the top 5. The graph acts as a third vote that lifts it into the context, and Recall@5 goes from 0.5 to 1.0.
- **q11: only half, with plain RRF.** The graph proposed 2 chunks, and only the runbook chunk, which dense and BM25 already ranked high, reached the top 5. The service-catalog chunk with Identity's lead and channel stayed out, because with plain RRF one vote from a retriever that proposes only 2 chunks weighs too little against chunks voted for by both other retrievers.
- **q12: no.** KST-6002 was not extracted, so no entity is found and the graph proposes nothing (section 7.7).
- **q02:** the KST-4012 node exists but has no edges, so the graph proposes nothing.

**What we changed, and the measured effect:** weighting the graph ×2. q11's service-catalog chunk then enters the top 5, as a graph-only chunk, so q11's Recall@5 goes from 0.5 to 1.0. Kestrel `two_hop` Recall@5 goes from 0.67 to 0.83, overall Recall@5 from 0.90 to 0.95, and no HitRate drops on any question type. More candidates (`--candidates 40`) would not help the graph, which never proposes more than 3 chunks here; a smaller RRF k would sharpen every retriever's top ranks, not only the graph's.

### 5.6 A question where adding a retriever made things worse

**Vellora V02** (split_answer): "How much water does the Vellora Duo tank hold, and what is the recommended minimum capacity of the container used beneath the outlets during descaling?"

- **BM25 alone:** both evidence items (the tank volume and the descaling container) are in the top 5. Recall@5 = 1.0.
- **Dense + BM25:** the first evidence stays #1 (dense #2, BM25 #1), but the descaling-container chunk falls out of the top 5. Recall@5 = 0.5.

**Why:** the question's wording ("capacity", "beneath") does not match the manual's ("at least 1.5 litres", "under"); V2's Words analysis already showed this. Dense therefore ranks the descaling chunk low, so after fusion it carries only BM25's vote. With plain RRF, chunks that both retrievers rank moderately outrank a chunk that only one retriever ranks high. Adding dense pushed a correct BM25 result out of the context.

**The same mechanism at k = 1:** on Kestrel q01 ("What does error KST-2044 mean…"), BM25 ranks the KST-2044 chunk #1. After fusion it is #3, behind a runbook chunk ranked #1 by dense and #3 by BM25.

**Weighted RRF can also make things worse:** on Kestrel q03 ("What happens, step by step, when a customer's card payment fails?", 4 evidence items), BM25 ×2 drops Recall@5 from 1.0 to 0.25. Most of the evidence is found by dense, and doubling BM25's vote pushes it out. This is why the default stays unweighted (5.4).

## 6. BM25

**Index.** An inverted index: for every word, the chunks that contain it and how many times. Each chunk also stores its length in tokens and its document. Search only visits the chunks that contain a question word. Scoring is Okapi BM25 with k1 = 1.5 and b = 0.75, and the IDF is `ln((N − n + 0.5) / (n + 0.5) + 1)`. The index is stored as JSON per corpus in `bm25_index/`, which is not committed because it can always be rebuilt from ChromaDB. In the API it is loaded once and then kept in memory.

**Tokenization.** BM25 uses the Words tab's tokenizer: lowercase; words are letters and digits, optionally joined by `'` or `-`; stopwords and one-letter words are dropped; no stemming. Keeping hyphenated tokens whole matters most for identifiers: `KST-5031` stays one token, `kst-5031`, which gets one of the highest IDFs in Kestrel. Because both features share the tokenizer, the Words tab shows exactly the words BM25 sees.

**Updates.** Upload, index, rebuild and delete all trigger a full rebuild of the corpus index from ChromaDB. A corpus rebuild triggers it once, not once per document. A missing or old-format file is rebuilt the first time BM25 is used.

**Measured behavior.**

- On the espresso manual alone, the E47 chunk ranks #1, reproducing the class example.
- On the whole Vellora corpus, which includes *Crime and Punishment*, the same question ranks the E47 chunk #4. The novel uses `mean` and `fix` often, which lowers their IDF, and a novel chunk takes #1.
- IDF is computed over the whole corpus even when the user selects only some documents. This is a deliberate trade-off: per-selection IDF would mean rebuilding statistics on every question.

**Limits.** The tokenizer only knows `a–z`, so accented names break: "Inès" becomes "in" + "s", and both are dropped. Kestrel's Messaging lead cannot be found by name with BM25.

## 7. Concept Graph (Kestrel)

### 7.1 Schema

8 entity types and 12 relations (D14). Every relation lists its allowed source and target types, and the schema is stored in the graph file itself.

### 7.2 Extraction and validation

One Gemini call per chunk (`gemini-3.5-flash-lite`, JSON mode, temperature 0). Validation as in D15. Every node and every edge points to the chunk(s) it came from, and every edge also stores the quote that supports it (provenance).

We ran the extraction three times, measuring each version:

| Run | New LLM calls | Tokens (prompt + completion) | Nodes | Edges | Rejected |
|---|---:|---:|---:|---:|---:|
| Prompt 1 | 43 | 28,453 + 3,454 | 38 | 19 | 5 |
| Prompt 1, corrected quote check | 0 (cache) | 0 | 38 | 24 | 0 |
| Prompt 2 (final graph) | 43 | 36,236 + 4,009 | 55 | 29 | 0 |

**What does not validate, and what happened.** With prompt 1, 5 triples were rejected with "quote not found in chunk". All 5 were rejected because of our check, not the model: it copied `**Escalate:**` as `Escalate:`, dropping the markdown, so the quote no longer matched. We made the check ignore `*` and `` ` `` and rebuilt from the cache at no cost. That recovered 5 triples, but 2 of them are wrong: KST-3012 → "Analytics Eng" (a team name cut off at the chunk boundary) and KST-3020 → Analytics Engine (KST-3012's escalation line, attached to the wrong error code). **The quote check proves the quote exists, not that the triple is right.**

**Why prompt 2.** Prompt 1 extracted mainly the service catalog: one edge from the runbook, none from the product guide or the billing policies, and no edges for q12. Prompt 2 asks for every fact, every table row and every "Escalate:" line, from the right error code, and tells the model to skip cut-off names. It added Payments Core → #oncall-payments, audit-log → Identity, KST-5031 escalates_to postroom and the plan → feature edges, and removed the wrong KST-3020 → Analytics Engine edge. The model still ignored the cut-off-name rule.

### 7.3 Entity resolution

- **Merge we got right:** `postroom` appears in 3 chunks of 2 documents (the runbook's KST-5031 section and the service catalog) and became one node. That merge is what links q08's first hop to its second. Platform (4 chunks) and Identity (3) were merged the same way.
- **Merges we missed:** "Analytics Engine" and "Payments Core" were each typed as a Team in some chunks and as a Service in others. Our resolution never merges across types, so each exists twice; both are listed under `possible_missed_merges`. "Analytics Eng" (a truncated name) is a third missed merge: it is a separate node, with no link to the Analytics Engine team.
- No two different spellings of the same entity were merged (0 spelling merges), because the model wrote each name the same way every time.

### 7.4 Retrieval

As in D17: whole-word entity matching, a walk of up to 3 hops in both directions, hubs not expanded, and chunk score Σ 1/hop. The response returns the entities found, the edges followed (with hop and quote), the graph's ranked chunks, and which of them reached the answer context. The UI shows all of these.

### 7.5 The graph in numbers

| | |
|---|---|
| Chunks | 43 |
| Nodes / edges | 55 / 29 |
| Merges (cross-chunk) / missed merges flagged | 9 nodes from several chunks / 2 |
| Final build | 43 calls, 36,236 + 4,009 tokens, 201 s (4 s pause between calls for the free tier) |
| All three runs | 86 calls, 64,689 + 7,463 tokens, 58.6 s of LLM time |
| Rejected / failed chunks (final) | 0 / 0 |
| Question time | 0 LLM calls, about 1.9 ms |

### 7.6 Extraction quality

**10 random triples** (fixed seed 2026, `graph/extraction_grades.csv`): **10 correct / 0 wrong / 0 unsupported.**

This overstates the quality. 22 of the 29 edges come from the service catalog, so 7 of the 10 sampled triples come from its team and service chunks. The two wrong triples we know of (section 7.2) were not drawn. One correct triple, KST-6010 → Payments Core, also loses its condition ("only if more than three customers are affected"), which the schema cannot store.

**How we think a concept graph should be evaluated:** sampling measures precision, so it must be paired with a **checklist** that measures recall: the facts the documents state, and how many the graph contains. For Kestrel:

| Fact family | In the documents | In the graph |
|---|---:|---:|
| Service owners (catalog) | 9 | 7 (auth and ledger missing) |
| Team leads and on-call channels (catalog table) | 12 | 12 |
| "Escalate: to X" lines (runbook) | 6 | 3 correct (KST-3012 points to the truncated node; KST-2040 and KST-6002 missing) |

**Precision is high; recall is the weakness.** A third measure is the one in section 5: whether the graph moves `two_hop` Recall.

### 7.7 Where the graph fails

- **q12:** KST-6002 was not extracted at all, so the question contains no known entity and the graph proposes nothing. q12 depends entirely on dense and BM25, whose Recall@5 stays at 0.5 (one hop out of two).
- **Isolated nodes:** KST-4012 (q02) exists as a node but has no edge, so finding it in the question gives the graph nothing to walk.
- **One bad triple and what it cost:** KST-3012 → "Analytics Eng". A question about KST-3012's escalation reaches a dead-end node: the walk cannot continue to the Analytics Engine team, its lead (Léa Fontaine) or its channel (#oncall-analytics).
- **False-positive entities:** Kestrel has a plan called "Team", so any question containing the word "team" also seeds the walk from the Team plan. This happened on q08 and q11. Here the plan has no edges, so it added nothing, but on a richer graph it would pull in plan-related chunks.
- **Schema limits:** the billing policies produce no edges at all, because the schema has no relations for refunds, grace periods or prices. The graph cannot help on `versions` questions, and conditions are lost.
- **KST-2xxx** became an ErrorCode node, although it is a code pattern, not a code.
- **Stale provenance:** edges point to chunk ids. After re-chunking with other settings, the graph must be rebuilt; the cache makes that free when the chunk text is unchanged.

### 7.8 Optional features

- **Aliases:** the model can list other names for an entity; these are merged with union-find and shown on the node.
- Not implemented: properties, time validity (from/until), a graph of our own corpus. For the Vellora manual, error codes, symptoms and fixes would fit a similar schema; for *Crime and Punishment*, the schema would have to be redesigned.

## 8. Words Analysis and Retrieval Diagnostics

The Words feature was used to investigate retrieval behavior in both corpora.

### Vellora — V02

**Question:** How much water does the Vellora Duo tank hold, and what is the recommended minimum capacity of the container used beneath the outlets during descaling?

The Words question analysis found that `capacity` and `beneath` were absent from the indexed vocabulary.

Inspection of the chunk locations showed that chunk 17 of `vellora-duo-espresso-machine.txt` contains the decisive evidence: a container of at least 1.5 litres must be placed under the coffee outlets during descaling.

The source uses "under" rather than "beneath", and expresses the capacity as a volume rather than using the word "capacity". This indicates a vocabulary mismatch that may contribute to the ranking miss, although the evidence is present in the corpus.

### Kestrel — q06

**Question:** Can I get a refund on my annual plan 45 days after it renewed?

The Words question analysis found `refund` (13 occurrences), `annual` (17), `days` (28) and `request` (8) in the indexed corpus, while `45` and `renewing` were absent.

Relevant vocabulary occurs in both `billing-policy-2025.md` and `billing-policy-2026.md`. In particular, chunks 3 and 4 of the 2026 policy contain overlapping refund, annual-plan and day-related vocabulary.

The presence of similar vocabulary in both policy versions may contribute to retrieval confusion. The question also uses paraphrased wording rather than the source's exact terminology. The generated answer's failure to prioritize the current policy is an additional version-handling limitation.

## 9. Known Failures and Limitations

**Updated in V3**

1. ~~Exact identifier matching is corpus-specific.~~ **Fixed:** the `E##` rule is retired. BM25 matches any identifier, `KST-` codes included.
2. **Version ambiguity** (unchanged). The graph cannot help: the billing policies produce no edges.
3. **Incomplete multi-hop retrieval:** now measurable with Recall@k. The graph completes q08 and q11, but not q12 (missing edge).
4. **Fusion lowers Hit@1** compared with BM25 alone (0.7 → 0.6 on both corpora). Weighting BM25 fixes it on Vellora and hurts Kestrel.
5. **Accented words** are split by the tokenizer ("Inès" → "in" + "s"), so BM25 cannot match them.
6. **Corpus-wide IDF:** selecting a subset of documents does not change the IDF, so common words in unselected documents still lower the weight of question words.
7. **Graph extraction recall:** service owners 7/9, escalations 3/6. The quote check cannot catch a quote attached to the wrong entity.
8. **Graph entity matching** is string-based: no paraphrases, and false positives on common words that are also entity names ("team").
9. **Graph only on Kestrel**, with a schema written for it.
10. **Error reporting** (V2 limitation, fixed): invalid documents or retrievers now return 400 with the reason, instead of a 502 "answer could not be validated".

**Unchanged from V2:** generation omissions (V06, q04), knowledge leakage without retrieval (V04, V05, q09), fixed-size chunking that cuts names and table rows (it produced the "Analytics Eng" node), non-transactional rebuilds, and small question sets (12 + 12).

## 10. Final Assessment

**V2.**

The V2 implementation successfully extends the original application to support two separate corpora and introduces a Words analysis feature for retrieval diagnosis.

Hybrid retrieval improves Vellora performance, especially for split-answer questions. Kestrel retrieval reaches 100% HitRate@10, but version ranking and incomplete multi-hop evidence remain important limitations.

The no-context baseline correctly answers only 2 of 18 answerable questions. With retrieval, 17 of 22 generated answers were graded correct and four partial. This demonstrates the value of grounding answers in the supplied knowledge bases, while also showing that retrieval quality and answer-generation quality must be evaluated separately.

Future improvements should prioritize corpus-aware identifier matching, current-version prioritization, complete evidence coverage for multi-hop questions and stricter answer completeness checks.

**V3.** BM25 fixes identifiers (Kestrel `exact_identifier` Hit@5 0.50 → 1.00). BM25 and RRF lift both corpora from the V2 baseline to Hit@3 = 1.0. Recall@5 goes from 0.85 to 0.95 on Vellora and from 0.75 to 0.85 on Kestrel, with no LLM call at question time and 3–8 ms of extra retrieval. On Kestrel, the concept graph raises `two_hop` evidence coverage: `two_hop` Recall@5 from 0.50 to 0.67, and 0.83 with graph ×2 (overall Recall@5 from 0.85 to 0.90 and 0.95). It cost 43 extraction calls and about 40k tokens, once. Its failures are specific and documented: one missing edge (q12), one truncated node, cross-type duplicates, and no coverage of billing facts. The main open problem is Hit@1 after fusion, and the experiments show that fixed weights are not the answer, because the best weights differ between the two corpora.

## 11. Deliverables

**V2**

- `evaluation/questions.vellora.jsonl`
- `evaluation/questions.kestrel.jsonl`
- `evaluation/no_context_vellora.jsonl`
- `evaluation/no_context_kestrel.jsonl`
- `evaluation/no_context_grading_v2.csv`
- `evaluation/results_v2_vellora_hybrid.jsonl`
- `evaluation/results_v2_vellora_semantic.jsonl`
- `evaluation/results_v2_kestrel_hybrid.jsonl`
- `evaluation/results_v2_kestrel_semantic.jsonl`
- `evaluation/summary_v2.csv`
- `evaluation/answers_v2_vellora_hybrid.jsonl`
- `evaluation/answers_v2_kestrel_hybrid.jsonl`
- `evaluation/grading_v2.csv`


The README documents how to start the application, select a corpus, run evaluations and use the Words feature.

**V3**

- `backend/tokenizer.py`, `backend/bm25.py`, `backend/graph_schema.py`, `backend/graph_build.py`, `backend/graph.py`, `backend/graph_sample.py`
- `graph/kestrel-graph.json`, `graph/cache/kestrel-extractions.jsonl`, `graph/extraction_grades.csv`
- `evaluation/questions.kestrel.jsonl` (q11 and q12 added, no existing question edited), `evaluation/no_context_kestrel.jsonl`
- `evaluation/results_v3_<corpus>_<configuration>.jsonl` (7 Kestrel and 4 Vellora configurations), `evaluation/summary_v3.csv`
- `docs-images/q08-graph-path.png`
- `README.md`: switching retrievers on and off, building the graph, running the evaluation.

## 12. Version Control

The V2 release is tagged `v2`. The V3 implementation was built in small commits (BM25, retrievers and RRF, API, UI, evaluation, graph extraction, quote-check fix, prompt 2, graph retriever, graph UI, weighted RRF and graph statistics, new questions, grading, README, report). The V3 release is tagged `v3`.
