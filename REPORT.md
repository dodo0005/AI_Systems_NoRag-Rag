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

## 5. Words Analysis and Retrieval Diagnostics

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

## 6. Known Failures and Limitations

1. **Exact identifier matching is corpus-specific.** The current exact-match enhancement recognizes error codes matching `E##`, but not Kestrel identifiers such as `KST-2044`. This explains why the two Kestrel retrieval configurations have identical results.

2. **Version ambiguity.** The 2025 and 2026 billing policies contain similar information. The generator does not consistently prioritize the current policy.

3. **Incomplete multi-hop retrieval.** Kestrel q08 demonstrates that finding the first part of an answer does not guarantee retrieval of the second part. A more complete evidence-coverage metric would help expose this problem.

4. **Generation omissions.** Some retrieved passages contain the required information, but the generated answer omits details, as seen in V06 and q04.

5. **Knowledge leakage.** General knowledge can answer some questions without retrieval, as demonstrated by V04 and V05. Unanswerable questions can also trigger confusion with unrelated products, as demonstrated by q09.

6. **Chunking limitations.** Fixed-size character chunks can split related information across passages and produce approximate line ranges.

7. **Index rebuilding.** A rebuild clears the selected collection before indexing. Transactional recovery is not implemented.

8. **Evaluation dataset size.** The datasets contain 12 Vellora questions and 10 Kestrel questions. Results are useful for comparing the tested configurations but should not be generalized to all possible queries.

## 7. Final Assessment

The V2 implementation successfully extends the original application to support two separate corpora and introduces a Words analysis feature for retrieval diagnosis.

Hybrid retrieval improves Vellora performance, especially for split-answer questions. Kestrel retrieval reaches 100% HitRate@10, but version ranking and incomplete multi-hop evidence remain important limitations.

The no-context baseline correctly answers only 2 of 18 answerable questions. With retrieval, 17 of 22 generated answers were graded correct and four partial. This demonstrates the value of grounding answers in the supplied knowledge bases, while also showing that retrieval quality and answer-generation quality must be evaluated separately.

Future improvements should prioritize corpus-aware identifier matching, current-version prioritization, complete evidence coverage for multi-hop questions and stricter answer completeness checks.

## 8. Deliverables

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

## 9. Version Control

The V2 implementation and evaluation artifacts are maintained in Git. The final V2 release is identified by a local `v2` tag after final verification.

No remote push is performed until explicitly agreed by the team.