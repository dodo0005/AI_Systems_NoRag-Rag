import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path

from backend.rag import retrieve_chunks

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "evaluation_set.csv"
RESULTS_DIR = ROOT / "evaluation"


def ask_api(api_url, question, document_id, top_k):
    """Call the existing FastAPI question endpoint."""
    payload = {
        "question": question,
        "document_ids": [document_id],
        "top_k": top_k,
    }

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        return None, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="current")
    args = parser.parse_args()

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    with DATASET.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))

    if not questions:
        raise SystemExit("The evaluation dataset is empty.")

    results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"].strip()
        document_id = item["document_id"].strip()
        expected_source_text = item.get("expected_source_text", "").strip()
        expected_status = item["expected_status"].strip()

        print(f"[{index}/{len(questions)}] {question}")

        retrieved_chunks = []
        retrieval_error = ""

        try:
            retrieved_chunks = retrieve_chunks(
                question=question,
                document_ids=[document_id],
                top_k=args.top_k,
            )
        except Exception as exc:
            retrieval_error = str(exc)

        # Use a stable source marker to judge retrieval.
        # Unanswerable questions have no expected source marker.
        if expected_source_text and not retrieval_error:
            retrieval_hit = any(
                expected_source_text.casefold() in chunk["text"].casefold()
                for chunk in retrieved_chunks
            )
        else:
            retrieval_hit = ""

        api_result, api_error = ask_api(
            args.api_url,
            question,
            document_id,
            args.top_k,
        )

        actual_status = ""
        answer = ""
        references = []

        if api_result is not None:
            actual_status = api_result.get("status", "")
            answer = api_result.get("answer", "")
            references = api_result.get("references", [])

        results.append(
            {
                "question": question,
                "document_id": document_id,
                "expected_answer": item["expected_answer"],
                "expected_source": item["expected_source"],
                "expected_source_text": expected_source_text,
                "top_k": args.top_k,
                "retrieved_chunk_ids": ";".join(
                    chunk["chunk_id"] for chunk in retrieved_chunks
                ),
                "retrieved_source_texts": " || ".join(
                    chunk["text"].replace("\n", " ") for chunk in retrieved_chunks
                ),
                "retrieval_hit": retrieval_hit,
                "expected_status": expected_status,
                "actual_status": actual_status,
                "status_match": (
                    actual_status == expected_status if actual_status else ""
                ),
                "answer": answer,
                "api_error": api_error,
                "retrieval_error": retrieval_error,
                "answer_correct_manual": "",
            }
        )

    output_file = RESULTS_DIR / f"results_{args.label}.csv"

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    applicable = [
        row
        for row in results
        if row["expected_source_text"] and row["retrieval_hit"] != ""
    ]
    hits = sum(row["retrieval_hit"] is True for row in applicable)

    status_checked = [row for row in results if row["actual_status"]]
    status_matches = sum(row["status_match"] is True for row in status_checked)

    print("\n--- Evaluation summary ---")
    print(f"Questions: {len(results)}")
    print(f"Top-k: {args.top_k}")

    if applicable:
        print(
            f"Retrieval Hit@{args.top_k}: "
            f"{hits}/{len(applicable)} = "
            f"{hits / len(applicable):.1%}"
        )
    else:
        print("Retrieval Hit rate: no answerable source cases to score")

    if status_checked:
        print(
            f"Expected-status accuracy: "
            f"{status_matches}/{len(status_checked)} = "
            f"{status_matches / len(status_checked):.1%}"
        )
    else:
        print("Expected-status accuracy: no successful API responses")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
