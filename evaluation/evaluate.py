"""V3 retrieval evaluation: several retriever configurations, both corpora.

Examples (from the project root):
    python evaluation/evaluate.py                         # every configuration, both corpora
    python evaluation/evaluate.py --corpus kestrel --config dense dense+bm25
    python evaluation/evaluate.py --corpus kestrel --config dense+bm25 --generate-answers
    python evaluation/evaluate.py --corpus kestrel --candidates 40   # measure a change
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import RateLimitError

from backend.rag import generate_grounded_answer, list_documents, retrieve

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

QUESTION_FILES = {
    "vellora": EVALUATION_DIR / "questions.vellora.jsonl",
    "kestrel": EVALUATION_DIR / "questions.kestrel.jsonl",
}

# Configuration name -> retrievers switched on (fused with RRF), optional RRF
# weights (weighted RRF), optional corpora restriction (the graph is Kestrel only).
# Weights were fixed before running, not tuned on the question sets.
CONFIGURATIONS = {
    "dense": {"retrievers": ["dense"]},
    "bm25": {"retrievers": ["bm25"]},
    "dense+bm25": {"retrievers": ["dense", "bm25"]},
    "dense+bm25_w": {
        "retrievers": ["dense", "bm25"],
        "weights": {"dense": 1.0, "bm25": 2.0},
    },
    "graph": {"retrievers": ["graph"], "corpora": ["kestrel"]},
    "dense+bm25+graph": {
        "retrievers": ["dense", "bm25", "graph"],
        "corpora": ["kestrel"],
    },
    "dense+bm25+graph_w": {
        "retrievers": ["dense", "bm25", "graph"],
        "weights": {"dense": 1.0, "bm25": 1.0, "graph": 2.0},
        "corpora": ["kestrel"],
    },
}

# Defaults of the app; change them from the command line to measure their effect.
OPTIONS = {"candidates": 20, "rrf_k": 60}

TOP_K = 10
K_VALUES = (1, 3, 5, 10)


def load_questions(path):
    questions = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc

            if "question" not in item:
                raise ValueError(f"Missing question in {path.name}, line {line_number}")

            if not isinstance(item.get("answerable"), bool):
                raise ValueError(
                    f"'answerable' must be true or false in "
                    f"{path.name}, line {line_number}"
                )

            if item["answerable"] and not item.get("evidence"):
                raise ValueError(
                    f"Answerable question has no evidence in "
                    f"{path.name}, line {line_number}"
                )

            questions.append(item)

    if not questions:
        raise ValueError(f"No questions found in {path}")

    return questions


def normalize(value):
    return " ".join(str(value).casefold().split())


def evidence_found(chunk, evidence):
    """True if the chunk comes from the evidence's source and contains its quote."""
    source = Path(chunk.get("document_id", "")).name.casefold()
    text = normalize(chunk.get("text", ""))

    expected_source = Path(evidence.get("source", "")).name.casefold()
    quote = normalize(evidence.get("quote", ""))

    source_matches = not expected_source or source == expected_source
    quote_matches = not quote or quote in text

    return source_matches and quote_matches


def hit(chunks, evidence_items):
    """HitRate: at least one evidence item is in the chunks."""
    return any(
        evidence_found(chunk, evidence)
        for chunk in chunks
        for evidence in evidence_items
    )


def recall(chunks, evidence_items):
    """Recall: fraction of the evidence items found in the chunks.

    For a two_hop question, 1.0 means every hop was retrieved.
    """
    found = sum(
        any(evidence_found(chunk, evidence) for chunk in chunks)
        for evidence in evidence_items
    )
    return found / len(evidence_items)


def evaluate_question(item, document_ids, corpus, config):
    question = item["question"].strip()

    settings = CONFIGURATIONS[config]

    start = time.perf_counter()
    retrieval = retrieve(
        question=question,
        document_ids=document_ids,
        top_k=TOP_K,
        corpus=corpus,
        retrievers=settings["retrievers"],
        weights=settings.get("weights"),
        candidates=OPTIONS["candidates"],
        rrf_k=OPTIONS["rrf_k"],
    )
    total_ms = round((time.perf_counter() - start) * 1000, 1)

    chunks = retrieval["chunks"]
    answerable = item["answerable"]
    evidence = item.get("evidence", [])

    hits = {}
    recalls = {}
    if answerable:
        for k in K_VALUES:
            hits[f"hit_at_{k}"] = hit(chunks[:k], evidence)
            recalls[f"recall_at_{k}"] = round(recall(chunks[:k], evidence), 4)

    # Did the graph's chunks survive fusion? (graph configurations only)
    graph_stats = None
    if retrieval["graph_path"] is not None:
        top5 = chunks[:5]
        graph_stats = {
            "entities_found": [e["name"] for e in retrieval["graph_path"]["entities"]],
            "candidates": len(retrieval["ranked_lists"]["graph"]),
            "in_top_5": sum("graph" in c["found_by"] for c in top5),
            "only_graph_in_top_5": sum(set(c["found_by"]) == {"graph"} for c in top5),
        }

    return {
        "id": item.get("id", ""),
        "corpus": corpus,
        "configuration": config,
        "weights": settings.get("weights"),
        "options": dict(OPTIONS),
        "graph": graph_stats,
        "retrievers": retrieval["retrievers"],
        "question": question,
        "type": item.get("type", item.get("question_type", "other")),
        "answerable": answerable,
        "expected_answer": item.get("expected_answer", ""),
        "evidence": evidence,
        "retrieval_hits": hits,
        "retrieval_recall": recalls,
        # Cost of retrieval: no LLM call at query time (embeddings run locally).
        "cost": {
            "llm_calls": 0,
            "tokens": 0,
            "latency_ms": {**retrieval["timings_ms"], "total": total_ms},
        },
        "retrieved_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "fused_rank": chunk["fused_rank"],
                "found_by": chunk["found_by"],
                "evidence": [
                    index
                    for index, item_evidence in enumerate(evidence)
                    if evidence_found(chunk, item_evidence)
                ],
                "text": chunk["text"],
            }
            for chunk in chunks
        ],
    }


def summarize(results, corpus, config):
    answerable = [r for r in results if r["answerable"]]
    unanswerable_count = sum(not r["answerable"] for r in results)

    groups = {"overall": answerable}
    for question_type in sorted({r["type"] for r in answerable}):
        groups[question_type] = [r for r in answerable if r["type"] == question_type]

    average_ms = round(
        sum(r["cost"]["latency_ms"]["total"] for r in results) / len(results), 1
    )

    rows = []
    for question_type, group in groups.items():
        row = {
            "corpus": corpus,
            "configuration": config,
            "question_type": question_type,
            "answerable_questions": len(group),
            "unanswerable_questions": (
                unanswerable_count if question_type == "overall" else ""
            ),
        }

        for k in K_VALUES:
            row[f"HitRate@{k}"] = (
                round(
                    sum(r["retrieval_hits"][f"hit_at_{k}"] for r in group) / len(group),
                    4,
                )
                if group
                else ""
            )
        for k in K_VALUES:
            row[f"Recall@{k}"] = (
                round(
                    sum(r["retrieval_recall"][f"recall_at_{k}"] for r in group)
                    / len(group),
                    4,
                )
                if group
                else ""
            )

        row["avg_retrieval_ms"] = average_ms if question_type == "overall" else ""
        row["llm_calls_per_question"] = 0 if question_type == "overall" else ""

        with_graph = [r for r in group if r.get("graph")]
        row["graph_entity_found_rate"] = (
            round(
                sum(bool(r["graph"]["entities_found"]) for r in with_graph)
                / len(with_graph),
                4,
            )
            if with_graph
            else ""
        )
        row["avg_graph_chunks_in_top5"] = (
            round(sum(r["graph"]["in_top_5"] for r in with_graph) / len(with_graph), 2)
            if with_graph
            else ""
        )
        row["avg_graph_only_chunks_in_top5"] = (
            round(
                sum(r["graph"]["only_graph_in_top_5"] for r in with_graph)
                / len(with_graph),
                2,
            )
            if with_graph
            else ""
        )
        rows.append(row)

    return rows


def run_suffix():
    """Non-default options go into file names, so runs never overwrite each other."""
    suffix = ""
    if OPTIONS["candidates"] != 20:
        suffix += f"_c{OPTIONS['candidates']}"
    if OPTIONS["rrf_k"] != 60:
        suffix += f"_k{OPTIONS['rrf_k']}"
    return suffix


def results_path(corpus, config):
    return EVALUATION_DIR / f"results_v3_{corpus}_{config}{run_suffix()}.jsonl"


def answers_path(corpus, config):
    return EVALUATION_DIR / f"answers_v3_{corpus}_{config}{run_suffix()}.jsonl"


def generate_answers(corpus, config):
    """Generate answers from saved retrieval results (resumable)."""
    input_path = results_path(corpus, config)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path.name} not found. Run the retrieval evaluation first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        results = [json.loads(line) for line in file if line.strip()]

    output_path = answers_path(corpus, config)
    completed_ids = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            completed_ids = {json.loads(line)["id"] for line in file if line.strip()}

    remaining = [r for r in results if r["id"] not in completed_ids]
    print(f"\n{corpus}/{config}: {len(completed_ids)} done, {len(remaining)} to go")

    with output_path.open("a", encoding="utf-8") as output:
        for index, result in enumerate(remaining, start=1):
            print(f"[{index}/{len(remaining)}] {result['id']}")

            for attempt in range(4):
                try:
                    start = time.perf_counter()
                    generated = generate_grounded_answer(
                        question=result["question"],
                        retrieved_chunks=result["retrieved_chunks"],
                    )
                    generation_ms = round((time.perf_counter() - start) * 1000, 1)
                    break
                except RateLimitError:
                    if attempt == 3:
                        raise
                    delay = 15 * (2**attempt)
                    print(f"Rate limit: waiting {delay}s...")
                    time.sleep(delay)

            generation = json.loads(generated.model_dump_json())
            generation["latency_ms"] = generation_ms

            output.write(
                json.dumps({**result, "generation": generation}, ensure_ascii=False)
                + "\n"
            )
            output.flush()

    print(f"Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="V3 retrieval evaluation.")
    parser.add_argument(
        "--corpus", choices=["all", "vellora", "kestrel"], default="all"
    )
    parser.add_argument(
        "--config",
        nargs="+",
        choices=list(CONFIGURATIONS),
        default=list(CONFIGURATIONS),
        help="Retriever configurations to run (default: all).",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Generate answers from saved retrieval results (calls the LLM).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Chunks each retriever proposes before fusion (default 20).",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60, help="RRF constant k (default 60)."
    )
    args = parser.parse_args()

    OPTIONS["candidates"] = args.candidates
    OPTIONS["rrf_k"] = args.rrf_k

    corpora = ["vellora", "kestrel"] if args.corpus == "all" else [args.corpus]

    def applies(config, corpus):
        return corpus in CONFIGURATIONS[config].get("corpora", [corpus])

    if args.generate_answers:
        for corpus in corpora:
            for config in args.config:
                if applies(config, corpus):
                    generate_answers(corpus, config)
        return

    summary_rows = []

    for corpus in corpora:
        questions = load_questions(QUESTION_FILES[corpus])
        document_ids = [
            d["document_id"] for d in list_documents(corpus) if d.get("indexed")
        ]

        if not document_ids:
            raise RuntimeError(f"No indexed documents in corpus '{corpus}'.")

        print(
            f"\n{corpus.upper()}: {len(questions)} questions, {len(document_ids)} documents"
        )

        for config in args.config:
            if not applies(config, corpus):
                continue
            print(f"  configuration: {config}")
            results = [
                evaluate_question(item, document_ids, corpus, config)
                for item in questions
            ]

            with results_path(corpus, config).open("w", encoding="utf-8") as file:
                for result in results:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")

            summary_rows.extend(summarize(results, corpus, config))

    summary_path = EVALUATION_DIR / f"summary_v3{run_suffix()}.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to {summary_path.name}\n")
    for row in summary_rows:
        if row["question_type"] == "overall":
            print(
                f"{row['corpus']:8} {row['configuration']:20} "
                f"Hit@1/3/5/10 = {row['HitRate@1']}/{row['HitRate@3']}/"
                f"{row['HitRate@5']}/{row['HitRate@10']}   "
                f"Recall@5 = {row['Recall@5']}   {row['avg_retrieval_ms']} ms"
            )


if __name__ == "__main__":
    main()
