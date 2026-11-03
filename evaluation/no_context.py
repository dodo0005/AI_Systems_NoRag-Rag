import json
import time
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI, RateLimitError

from backend.rag import GEMINI_MODEL, GOOGLE_API_KEY

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"

QUESTION_SETS = {
    "vellora": EVALUATION_DIR / "questions.vellora.jsonl",
    "kestrel": EVALUATION_DIR / "questions.kestrel.jsonl",
}

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 15


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_no_context(corpus: str, input_path: Path) -> Path:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not configured. Check your .env file.")

    client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    questions = read_jsonl(input_path)
    output_path = EVALUATION_DIR / f"no_context_{corpus}.jsonl"

    # Preserve existing answers so interrupted runs can resume.
    completed_ids = set()

    if output_path.exists():
        existing_results = read_jsonl(output_path)
        completed_ids = {
            item["id"]
            for item in existing_results
            if item.get("no_context_answer") is not None
        }

    remaining = [item for item in questions if item["id"] not in completed_ids]

    print(
        f"[{corpus}] {len(completed_ids)} already completed; "
        f"{len(remaining)} remaining."
    )

    with output_path.open("a", encoding="utf-8") as output:
        for index, item in enumerate(questions, start=1):
            if item["id"] in completed_ids:
                continue

            question = item["question"]

            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=GEMINI_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer the user's question using your "
                                    "general knowledge only. You have not "
                                    "been given any documents or retrieved "
                                    "context. Do not claim that you consulted "
                                    "a knowledge base. If you do not know, "
                                    "say so."
                                ),
                            },
                            {"role": "user", "content": question},
                        ],
                    )

                    answer = response.choices[0].message.content or ""
                    break

                except RateLimitError:
                    if attempt == MAX_RETRIES - 1:
                        raise

                    delay = INITIAL_RETRY_DELAY * (2**attempt)

                    print(
                        f"[{corpus}] Rate limit for {item['id']}. "
                        f"Waiting {delay}s before retry "
                        f"{attempt + 2}/{MAX_RETRIES}..."
                    )

                    time.sleep(delay)

            result = {
                **item,
                "no_context_answer": answer,
                "no_context_model": GEMINI_MODEL,
                "no_context_correct": None,
            }

            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()

            completed_ids.add(item["id"])

            print(
                f"[{corpus}] {len(completed_ids)}/{len(questions)} "
                f"completed: {item['id']}"
            )

    print(f"Saved: {output_path.relative_to(ROOT)}")
    return output_path


def main() -> None:
    for corpus, input_path in QUESTION_SETS.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Question set not found: {input_path}")

        run_no_context(corpus, input_path)


if __name__ == "__main__":
    main()
