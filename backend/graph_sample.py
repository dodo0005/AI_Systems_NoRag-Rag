"""Extraction quality: pick 10 triples at random for grading by hand.

    python -m backend.graph_sample              # writes graph/extraction_grades.csv
    python -m backend.graph_sample --summarize  # counts your grades

Open the CSV, read each chunk_text, and fill the "grade" column with:
  correct      the chunk validates the triple
  wrong        the chunk says something else (entity, relation or direction)
  unsupported  nothing in the chunk says it (inferred or invented)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter

from backend.graph import graph_file
from backend.rag import PROJECT_ROOT, get_collection

GRADES_FILE = PROJECT_ROOT / "graph" / "extraction_grades.csv"
SEED = 2026  # fixed, so the same 10 triples come out every time
SAMPLE_SIZE = 10
GRADES = ("correct", "wrong", "unsupported")


def write_sample(corpus: str) -> None:
    if GRADES_FILE.exists():
        raise SystemExit(f"{GRADES_FILE.name} exists already: keep your grades, or delete it first.")

    graph = json.loads(graph_file(corpus).read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = random.Random(SEED).sample(graph["edges"], min(SAMPLE_SIZE, len(graph["edges"])))

    chunk_ids = [edge["evidence"][0]["chunk_id"] for edge in edges]
    stored = get_collection(corpus).get(ids=list(dict.fromkeys(chunk_ids)), include=["documents"])
    text_of = dict(zip(stored["ids"], stored["documents"]))

    with GRADES_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["n", "edge_id", "source", "relation", "target", "document_id",
             "chunk_id", "quote", "chunk_text", "grade", "comment"]
        )
        for number, edge in enumerate(edges, start=1):
            evidence = edge["evidence"][0]
            writer.writerow(
                [
                    number,
                    edge["id"],
                    nodes[edge["source"]]["name"],
                    edge["relation"],
                    nodes[edge["target"]]["name"],
                    evidence["document_id"],
                    evidence["chunk_id"],
                    evidence["quote"],
                    text_of.get(evidence["chunk_id"], "(chunk not found: rebuild the graph)"),
                    "",
                    "",
                ]
            )

    print(f"Wrote {SAMPLE_SIZE} triples to {GRADES_FILE.relative_to(PROJECT_ROOT)} (seed {SEED}).")


def summarize() -> None:
    with GRADES_FILE.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    counts = Counter(row["grade"].strip().lower() for row in rows)
    missing = [row["n"] for row in rows if row["grade"].strip().lower() not in GRADES]
    print(" / ".join(f"{counts[g]} {g}" for g in GRADES))
    if missing:
        print("Not graded yet (or misspelled):", ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Sample graph triples for grading.")
    parser.add_argument("--corpus", default="kestrel", choices=["kestrel"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize() if args.summarize else write_sample(args.corpus)


if __name__ == "__main__":
    main()
