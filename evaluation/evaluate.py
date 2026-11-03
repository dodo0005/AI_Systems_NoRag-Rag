import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import time
from openai import RateLimitError

from backend.rag import (
    generate_grounded_answer,
    get_collection,
    list_documents,
    retrieve_chunks,
)

ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT / "evaluation"

QUESTION_FILES = {
    "vellora": EVALUATION_DIR / "questions.vellora.jsonl",
    "kestrel": EVALUATION_DIR / "questions.kestrel.jsonl",
}

CONFIGURATIONS = {
    "hybrid": True,
    "semantic": False,
}

TOP_K = 10
HIT_K_VALUES = (1, 3, 5, 10)


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


def chunk_source(chunk):
    return (
        chunk.get("document_id") or chunk.get("filename") or chunk.get("source") or ""
    )


def evidence_found(chunk, evidence_items):
    source = Path(chunk_source(chunk)).name.casefold()
    text = normalize(chunk.get("text", ""))

    for evidence in evidence_items:
        expected_source = Path(evidence.get("source", "")).name.casefold()
        quote = normalize(evidence.get("quote", ""))

        source_matches = not expected_source or source == expected_source
        quote_matches = not quote or quote in text

        if source_matches and quote_matches:
            return True

    return False


def question_hit(chunks, evidence):
    return any(evidence_found(chunk, evidence) for chunk in chunks)


def evaluate_question(item, document_ids, corpus, config, use_exact_match):
    question = item["question"].strip()

    chunks = retrieve_chunks(
        question=question,
        document_ids=document_ids,
        top_k=TOP_K,
        corpus=corpus,
        use_exact_match=use_exact_match,
    )

    answerable = item["answerable"]
    evidence = item.get("evidence", [])

    hits = {}
    if answerable:
        for k in HIT_K_VALUES:
            hits[f"hit_at_{k}"] = question_hit(chunks[:k], evidence)

    return {
        "id": item.get("id", ""),
        "corpus": corpus,
        "configuration": config,
        "question": question,
        "type": item.get("type", item.get("question_type", "other")),
        "answerable": answerable,
        "expected_answer": item.get("expected_answer", ""),
        "evidence": evidence,
        "retrieval_hits": hits,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk_source(chunk),
                "start_line": chunk.get("start_line", 1),
                "end_line": chunk.get("end_line", 1),
                "text": chunk.get("text", ""),
            }
            for chunk in chunks
        ],
    }


def summarize(results, corpus, config):
    relevant = [
        result
        for result in results
        if result["corpus"] == corpus and result["configuration"] == config
    ]

    answerable = [result for result in relevant if result["answerable"]]
    unanswerable_count = sum(not result["answerable"] for result in relevant)

    groups = {"overall": answerable}

    for question_type in sorted({r["type"] for r in answerable}):
        groups[question_type] = [r for r in answerable if r["type"] == question_type]

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

        for k in HIT_K_VALUES:
            hits = sum(result["retrieval_hits"][f"hit_at_{k}"] for result in group)
            row[f"HitRate@{k}"] = round(hits / len(group), 4) if group else ""

        rows.append(row)

    return rows


def generate_answers_from_saved_results(corpus):
    """Generate answers from saved retrieval results, supporting resume."""
    input_path = EVALUATION_DIR / f"results_v2_{corpus}_hybrid.jsonl"
    output_path = EVALUATION_DIR / f"answers_v2_{corpus}_hybrid.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Retrieval results not found: {input_path}. "
            "Run retrieval evaluation first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        results = [json.loads(line) for line in file if line.strip()]

    # Recover original line metadata from Chroma.
    chunk_ids = list(
        {
            chunk["chunk_id"]
            for result in results
            for chunk in result["retrieved_chunks"]
        }
    )

    collection = get_collection(corpus)
    stored_chunks = collection.get(
        ids=chunk_ids,
        include=["metadatas"],
    )

    metadata_by_id = {
        chunk_id: metadata
        for chunk_id, metadata in zip(
            stored_chunks["ids"],
            stored_chunks["metadatas"],
        )
    }

    for result in results:
        for chunk in result["retrieved_chunks"]:
            metadata = metadata_by_id.get(chunk["chunk_id"])

            if metadata is None:
                raise RuntimeError(
                    f"Missing Chroma metadata for chunk {chunk['chunk_id']}"
                )

            chunk["start_line"] = metadata.get("start_line", 1)
            chunk["end_line"] = metadata.get("end_line", 1)
            chunk["document_id"] = metadata["document_id"]

    # Resume from existing generated answers.
    completed_ids = set()

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            existing = [json.loads(line) for line in file if line.strip()]

        completed_ids = {
            item["id"] for item in existing if item.get("generation") is not None
        }

    remaining = [result for result in results if result["id"] not in completed_ids]

    print(
        f"\n{corpus.upper()}: {len(completed_ids)} already completed; "
        f"{len(remaining)} remaining"
    )

    with output_path.open("a", encoding="utf-8") as output:
        for index, result in enumerate(remaining, start=1):
            print(
                f"[{index}/{len(remaining)}] " f"Generating answer for {result['id']}"
            )

            generated = None

            for attempt in range(4):
                try:
                    generated = generate_grounded_answer(
                        question=result["question"],
                        retrieved_chunks=result["retrieved_chunks"],
                    )
                    break

                except RateLimitError:
                    if attempt == 3:
                        raise

                    delay = 15 * (2**attempt)
                    print(
                        f"Gemini rate limit. Waiting {delay}s "
                        f"before retry {attempt + 2}/4..."
                    )
                    time.sleep(delay)

            generation = json.loads(generated.model_dump_json())

            answer_result = {
                **result,
                "generation": generation,
            }

            output.write(json.dumps(answer_result, ensure_ascii=False) + "\n")
            output.flush()

            completed_ids.add(result["id"])

            print(f"Saved {result['id']} " f"({len(completed_ids)}/{len(results)})")

    print(f"Saved generated answers: {output_path.name}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval across Vellora and Kestrel."
    )
    parser.add_argument(
        "--corpus",
        choices=["all", "vellora", "kestrel"],
        default="all",
    )
    parser.add_argument(
        "--config",
        choices=["both", "hybrid", "semantic"],
        default="both",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Generate answers from saved hybrid retrieval results.",
    )
    args = parser.parse_args()
    if args.generate_answers:
        corpora = (
            ["vellora", "kestrel"]
            if args.corpus == "all"
            else [args.corpus]
        )

        for corpus in corpora:
            generate_answers_from_saved_results(corpus)

        return

    corpora = ["vellora", "kestrel"] if args.corpus == "all" else [args.corpus]

    configs = ["hybrid", "semantic"] if args.config == "both" else [args.config]

    all_results = []
    summary_rows = []

    for corpus in corpora:
        question_file = QUESTION_FILES[corpus]
        questions = load_questions(question_file)

        indexed_documents = [
            document["document_id"]
            for document in list_documents(corpus)
            if document.get("indexed")
        ]

        if not indexed_documents:
            raise RuntimeError(f"No indexed documents found for corpus '{corpus}'.")

        print(
            f"\n{corpus.upper()}: {len(questions)} questions, "
            f"{len(indexed_documents)} indexed documents"
        )

        for config in configs:
            use_exact_match = CONFIGURATIONS[config]
            print(f"\nRunning configuration: {config}")

            config_results = []

            for index, item in enumerate(questions, start=1):
                print(
                    f"[{index}/{len(questions)}] "
                    f"{item.get('id', '')}: {item['question']}"
                )

                result = evaluate_question(
                    item=item,
                    document_ids=indexed_documents,
                    corpus=corpus,
                    config=config,
                    use_exact_match=use_exact_match,
                )

                config_results.append(result)
                all_results.append(result)

            output_path = EVALUATION_DIR / f"results_v2_{corpus}_{config}.jsonl"

            with output_path.open("w", encoding="utf-8") as file:
                for result in config_results:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")

            summary_rows.extend(summarize(config_results, corpus, config))

            print(f"Saved detailed results: {output_path.name}")

    summary_path = EVALUATION_DIR / "summary_v2.csv"

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())

        with summary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nEvaluation complete.")
    print(f"Summary saved to: {summary_path}")

    for row in summary_rows:
        if row["question_type"] == "overall":
            print(
                f"{row['corpus']} / {row['configuration']}: "
                f"answerable={row['answerable_questions']}, "
                f"unanswerable={row['unanswerable_questions']}, "
                f"Hit@1={row['HitRate@1']}, "
                f"Hit@3={row['HitRate@3']}, "
                f"Hit@5={row['HitRate@5']}, "
                f"Hit@10={row['HitRate@10']}"
            )


if __name__ == "__main__":
    main()
