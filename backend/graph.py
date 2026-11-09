"""The concept graph as a retriever: question -> ranked list of chunk ids.

Strategy: local walk.
1. Find the question's entities: node names and aliases that appear, as whole
   words, in the question (string matching, no LLM call at question time).
2. Walk the graph from them, up to MAX_HOPS edges, in both directions.
   Hub nodes (too many edges) are not expanded: they connect to everything.
3. Every edge followed brings the chunks it was extracted from. A chunk scores
   1/hop for each edge that points to it: closer facts rank higher.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from backend.graph_schema import normalize_name

GRAPH_DIR = Path(__file__).resolve().parent.parent / "graph"
MAX_HOPS = 3
HUB_DEGREE = 12  # nodes with more edges than this are not expanded
MIN_NAME_LENGTH = 3
MAX_PATH_EDGES = 40  # edges returned for display


def graph_file(corpus: str) -> Path:
    return GRAPH_DIR / f"{corpus}-graph.json"


def match_form(text: str) -> str:
    """Lowercase, punctuation to spaces (keeping - and #), padded with spaces."""
    text = re.sub(r"[^\w#\-]+", " ", text.casefold())
    return f" {' '.join(text.split())} "


class ConceptGraph:
    def __init__(self, data: dict):
        self.nodes = {node["id"]: node for node in data["nodes"]}
        self.edges = data["edges"]
        self.metadata = data.get("metadata", {})

        self.adjacency: dict[str, list[dict]] = defaultdict(list)
        for edge in self.edges:
            self.adjacency[edge["source"]].append(edge)
            self.adjacency[edge["target"]].append(edge)

        # Every name a node is known by, in matching form.
        self.names: list[tuple[str, str]] = []
        for node in self.nodes.values():
            variants = {node["name"], *node.get("aliases", [])}
            for variant in variants:
                form = match_form(normalize_name(variant, node["type"])).strip()
                if len(form) >= MIN_NAME_LENGTH:
                    self.names.append((form, node["id"]))
        # Longest names first, so "payments core" wins over "core".
        self.names.sort(key=lambda item: -len(item[0]))

    @classmethod
    def load(cls, path: Path) -> "ConceptGraph":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def find_entities(self, question: str) -> list[str]:
        text = match_form(question)
        found: list[str] = []

        for form, node_id in self.names:
            needle = f" {form} "
            if needle in text and node_id not in found:
                found.append(node_id)
                # Blank the match so a shorter name cannot match inside it.
                text = text.replace(needle, " " + "_" * len(form) + " ")

        return found

    def search(
        self,
        question: str,
        allowed_document_ids: set[str] | None = None,
        n: int = 20,
        max_hops: int = MAX_HOPS,
    ) -> tuple[list[str], dict]:
        """Return (ranked chunk ids, path) for a question."""
        seeds = self.find_entities(question)

        visited = set(seeds)
        frontier = list(seeds)
        followed: list[tuple[dict, int]] = []
        seen_edges = set()

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node_id in frontier:
                is_hub = len(self.adjacency[node_id]) > HUB_DEGREE
                if is_hub and hop > 1:
                    continue  # only a seed hub is expanded, and only one hop

                for edge in self.adjacency[node_id]:
                    if edge["id"] in seen_edges:
                        continue
                    seen_edges.add(edge["id"])
                    followed.append((edge, hop))

                    neighbour = (
                        edge["target"] if edge["source"] == node_id else edge["source"]
                    )
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.append(neighbour)
            frontier = next_frontier

        scores: dict[str, float] = defaultdict(float)
        best_hop: dict[str, int] = {}
        path_edges = []

        for edge, hop in followed:
            chunk_ids = [
                evidence["chunk_id"]
                for evidence in edge["evidence"]
                if allowed_document_ids is None
                or evidence["document_id"] in allowed_document_ids
            ]
            if not chunk_ids:
                continue

            for chunk_id in chunk_ids:
                scores[chunk_id] += 1 / hop
                best_hop[chunk_id] = min(hop, best_hop.get(chunk_id, hop))

            path_edges.append(
                {
                    "source": self.nodes[edge["source"]]["name"],
                    "relation": edge["relation"],
                    "target": self.nodes[edge["target"]]["name"],
                    "hop": hop,
                    "chunk_ids": chunk_ids,
                    "quote": edge["evidence"][0]["quote"],
                }
            )

        ranked = sorted(scores, key=lambda c: (-scores[c], best_hop[c], c))[:n]
        path_edges.sort(key=lambda e: e["hop"])

        path = {
            "entities": [
                {
                    "name": self.nodes[node_id]["name"],
                    "type": self.nodes[node_id]["type"],
                }
                for node_id in seeds
            ],
            "edges": path_edges[:MAX_PATH_EDGES],
            "chunk_ids": ranked,
        }
        return ranked, path
