"""Final re-run next to V2 and V3: one table of the overall rows.

    python evaluation/compare_releases.py      # writes evaluation/comparison_v4.csv

Reads summary_v2.csv, summary_v3.csv and summary_v4.csv (missing files are skipped).
V2 only has HitRate@k, so its Recall columns stay empty.
"""

import csv
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
COLUMNS = ["release", "corpus", "configuration", "HitRate@1", "HitRate@3", "HitRate@5",
           "HitRate@10", "Recall@5", "Recall@10", "avg_retrieval_ms", "fingerprint"]


def main():
    rows = []
    for release in ("v2", "v3", "v4"):
        path = EVALUATION_DIR / f"summary_{release}.csv"
        if not path.exists():
            print(f"skipped: {path.name} not found")
            continue
        with path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("question_type") == "overall":
                    rows.append({"release": release, **{c: row.get(c, "") for c in COLUMNS[1:]}})

    rows.sort(key=lambda r: (r["corpus"], r["configuration"], r["release"]))
    output = EVALUATION_DIR / "comparison_v4.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['release']}  {r['corpus']:8} {r['configuration']:20} "
              f"Hit@5={r['HitRate@5']:7} Recall@5={r['Recall@5']:7} {r['avg_retrieval_ms']} ms")
    print(f"\nSaved {output.name}")


if __name__ == "__main__":
    main()
