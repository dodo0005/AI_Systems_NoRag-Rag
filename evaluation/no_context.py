import json
import time
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import llm

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"

QUESTION_SETS = {
    "vellora": EVALUATION_DIR / "questions.vellora.jsonl",
    "kestrel": EVALUATION_DIR / "questions.kestrel.jsonl",
}

MAX_RETRIES = 5


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_no_context(corpus: str, input_path: Path) -> Path:
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

            # backend/llm.py backs off and retries on rate limits.
            reply = llm.chat(
                question,
                system=(
                    "Answer the user's question using your "
                    "general knowledge only. You have not "
                    "been given any documents or retrieved "
                    "context. Do not claim that you consulted "
                    "a knowledge base. If you do not know, "
                    "say so."
                ),
                json_mode=False,
                temperature=None,
                max_attempts=MAX_RETRIES,
            )
            answer = reply.content

            result = {
                **item,
                "no_context_answer": answer,
                "no_context_model": reply.model,
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
