"""Build the concept graph of a corpus.

    python -m backend.graph_build --corpus kestrel

1. Extraction: one LLM call per chunk returns entities and (entity, relation,
   entity) triples, each with a quote copied from the chunk.
2. Validation: every entity and relation is checked with Pydantic against the
   schema, then against the relation's allowed types, then the quote must really
   be in the chunk. Invalid items are dropped and counted. If the whole answer is
   not valid JSON, the call is retried once with the error, then the chunk is skipped.
3. Entity resolution: names are normalised ("The Messaging team" -> "messaging");
   same type + same normalised name, or an alias given by the LLM, means same entity.
4. Storage: graph/<corpus>-graph.json, every node and edge pointing to its chunks.

LLM answers are cached per chunk text (graph/cache/), so a crash or a rebuild
never pays twice for the same chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from openai import OpenAI, RateLimitError
from pydantic import ValidationError

from backend.graph_schema import (
    ENTITY_TYPES,
    RELATION_SIGNATURES,
    ExtractedEntity,
    ExtractedRelation,
    normalize_name,
    normalize_text,
)
from backend.rag import GEMINI_MODEL, PROJECT_ROOT, get_collection

GRAPH_DIR = PROJECT_ROOT / "graph"
CACHE_DIR = GRAPH_DIR / "cache"
PROMPT_VERSION = "v1"

GRAPH_MODEL = os.getenv("GRAPH_MODEL", GEMINI_MODEL)
CALL_DELAY_SECONDS = float(os.getenv("GRAPH_CALL_DELAY", "4"))  # free tier: ~15 calls/min


def api_keys() -> list[str]:
    """GOOGLE_API_KEYS="key1,key2" lets the build switch key when one is rate-limited."""
    keys = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
    if not keys and os.getenv("GOOGLE_API_KEY"):
        keys = [os.getenv("GOOGLE_API_KEY")]
    if not keys:
        raise RuntimeError("Set GOOGLE_API_KEY (or GOOGLE_API_KEYS) in your .env file.")
    return keys


def build_prompt(document_id: str, text: str) -> str:
    entity_types = ", ".join(ENTITY_TYPES)
    relations = "\n".join(
        f'- {name}: {"/".join(sorted(src))} -> {"/".join(sorted(tgt))} ({meaning})'
        for name, (src, tgt, meaning) in RELATION_SIGNATURES.items()
    )

    return f"""You extract a knowledge graph from one excerpt of a company document.

Entity types (use only these): {entity_types}

Relation types (use only these, with these source -> target types):
{relations}

Rules:
- Extract only facts the excerpt states explicitly. Never use outside knowledge.
- Use entity names exactly as written in the excerpt (e.g. "KST-5031", "postroom",
  "Messaging", "#oncall-messaging").
- "aliases": other names the excerpt uses for the SAME entity (may be empty).
- Every relation's source and target must be names from your entities list.
- "quote": copy, word for word, the shortest part of the excerpt that states the relation.
- If the excerpt states no relation, return empty lists.

Return only a JSON object:
{{"entities": [{{"name": "...", "type": "...", "aliases": []}}],
  "relations": [{{"source": "...", "relation": "...", "target": "...", "quote": "..."}}]}}

DOCUMENT: {document_id}
EXCERPT:
{text}
"""


# ---------- LLM calls, with cache, retries and key rotation ----------


class Extractor:
    def __init__(self, corpus: str):
        self.keys = api_keys()
        self.key_index = 0
        self.cache_path = CACHE_DIR / f"{corpus}-extractions.jsonl"
        self.cache: dict[str, dict] = {}
        self.stats = Counter()

        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry

    def client(self) -> OpenAI:
        return OpenAI(
            api_key=self.keys[self.key_index],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def call(self, prompt: str) -> tuple[str, dict]:
        """One LLM call. On rate limits: wait, retry, then switch to the next key."""
        for key_round in range(len(self.keys)):
            for attempt in range(4):
                try:
                    started = time.perf_counter()
                    response = self.client().chat.completions.create(
                        model=GRAPH_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                    self.stats["llm_calls"] += 1
                    self.stats["prompt_tokens"] += usage["prompt_tokens"]
                    self.stats["completion_tokens"] += usage["completion_tokens"]
                    time.sleep(CALL_DELAY_SECONDS)
                    return response.choices[0].message.content or "", usage

                except RateLimitError:
                    self.stats["rate_limited"] += 1
                    delay = 15 * (2**attempt)
                    print(f"    rate limited, waiting {delay}s...")
                    time.sleep(delay)

            if len(self.keys) > 1:
                self.key_index = (self.key_index + 1) % len(self.keys)
                print(f"    switching to API key #{self.key_index + 1}")

        raise RuntimeError("Every API key is rate-limited. Re-run later: done chunks are cached.")

    def extract(self, chunk: dict) -> tuple[dict | None, str]:
        """Return (parsed JSON or None, status) for one chunk."""
        key = hashlib.sha256(
            f"{PROMPT_VERSION}|{GRAPH_MODEL}|{chunk['text']}".encode("utf-8")
        ).hexdigest()

        if key in self.cache:
            self.stats["cached_chunks"] += 1
            raw = self.cache[key]["raw"]
            parsed = parse_json(raw)
            return parsed, "cached" if parsed is not None else "failed"

        prompt = build_prompt(chunk["document_id"], chunk["text"])
        raw, usage = self.call(prompt)
        parsed = parse_json(raw)

        if parsed is None:
            # One retry, telling the model what was wrong.
            self.stats["retries"] += 1
            raw, usage = self.call(
                prompt
                + "\nYour previous answer was not a valid JSON object with "
                '"entities" and "relations" lists. Return only that JSON object.'
            )
            parsed = parse_json(raw)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, "chunk_id": chunk["chunk_id"], "raw": raw, "usage": usage}
        with self.cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.cache[key] = entry

        return parsed, "extracted" if parsed is not None else "failed"


def parse_json(raw: str) -> dict | None:
    """The answer must be a JSON object with two lists; anything else is a failure."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("entities"), list) or not isinstance(data.get("relations"), list):
        return None
    return data


# ---------- validation ----------


def validate_chunk_output(data: dict, chunk: dict, rejected: Counter, examples: list) -> tuple[list, list]:
    """Keep only the entities and relations that pass every check."""

    def reject(reason: str, item):
        rejected[reason] += 1
        if len(examples) < 25:
            examples.append({"chunk_id": chunk["chunk_id"], "reason": reason, "item": item})

    entities: list[ExtractedEntity] = []
    for item in data["entities"]:
        try:
            entities.append(ExtractedEntity.model_validate(item))
        except ValidationError:
            reject("entity fails schema", item)

    # name in this chunk -> entity types it was given
    types_by_name: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        for name in [entity.name, *entity.aliases]:
            types_by_name[normalize_text(name)].add(entity.type)

    chunk_text = normalize_text(chunk["text"])
    relations: list[tuple[ExtractedRelation, str, str]] = []

    for item in data["relations"]:
        try:
            relation = ExtractedRelation.model_validate(item)
        except ValidationError:
            reject("relation fails schema", item)
            continue

        source_types = types_by_name.get(normalize_text(relation.source), set())
        target_types = types_by_name.get(normalize_text(relation.target), set())
        if not source_types or not target_types:
            reject("relation uses an unknown entity", item)
            continue

        allowed_sources, allowed_targets, _ = RELATION_SIGNATURES[relation.relation]
        source_type = next(iter(sorted(source_types & allowed_sources)), None)
        target_type = next(iter(sorted(target_types & allowed_targets)), None)
        if source_type is None or target_type is None:
            reject("relation has wrong entity types", item)
            continue

        if normalize_text(relation.quote) not in chunk_text:
            reject("quote not found in chunk", item)
            continue

        relations.append((relation, source_type, target_type))

    return entities, relations


# ---------- entity resolution ----------


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def slug(text: str) -> str:
    return re.sub(r"[^\w#]+", "-", text).strip("-")


def resolve_and_assemble(extractions: list[tuple[dict, list, list]]) -> dict:
    """Merge entities across chunks and build nodes, edges and the merge log."""
    union = UnionFind()
    surface_names: dict[tuple, Counter] = defaultdict(Counter)
    chunks_of_key: dict[tuple, set] = defaultdict(set)
    alias_links: list[tuple] = []

    for chunk, entities, _relations in extractions:
        for entity in entities:
            key = (entity.type, normalize_name(entity.name, entity.type))
            union.find(key)
            surface_names[key][entity.name.strip()] += 1
            chunks_of_key[key].add(chunk["chunk_id"])

            for alias in entity.aliases:
                alias_key = (entity.type, normalize_name(alias, entity.type))
                if alias_key[1] and alias_key != key:
                    union.union(key, alias_key)
                    surface_names[alias_key][alias.strip()] += 1
                    alias_links.append((key, alias_key, chunk["chunk_id"]))

    # Group keys by their root: one group = one node.
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for key in list(union.parent):
        groups[union.find(key)].append(key)

    nodes, node_id_of_key, merges = [], {}, []
    for root, keys in sorted(groups.items()):
        names = Counter()
        chunk_ids = set()
        for key in keys:
            names.update(surface_names[key])
            chunk_ids |= chunks_of_key[key]

        # Display name: the most frequent spelling, the shortest one on ties.
        name = max(names, key=lambda n: (names[n], -len(n))) if names else root[1]
        node_id = f"{root[0].lower()}:{slug(root[1])}"
        for key in keys:
            node_id_of_key[key] = node_id

        nodes.append(
            {
                "id": node_id,
                "name": name,
                "type": root[0],
                "aliases": sorted(n for n in names if n != name),
                "chunk_ids": sorted(chunk_ids),
            }
        )

        if len(names) > 1:
            merges.append(
                {
                    "node": node_id,
                    "merged_names": sorted(names),
                    "method": "alias given by the LLM" if len(keys) > 1 else "same normalised name",
                }
            )

    # Same normalised name, different types: kept apart, listed for review.
    by_name: dict[str, set[str]] = defaultdict(set)
    for entity_type, name in union.parent:
        by_name[name].add(entity_type)
    possible_missed = [
        {"name": name, "types": sorted(types)}
        for name, types in sorted(by_name.items())
        if len(types) > 1
    ]

    edges_by_triple: dict[tuple, dict] = {}
    for chunk, _entities, relations in extractions:
        for relation, source_type, target_type in relations:
            source = node_id_of_key.get((source_type, normalize_name(relation.source, source_type)))
            target = node_id_of_key.get((target_type, normalize_name(relation.target, target_type)))
            if not source or not target or source == target:
                continue

            triple = (source, relation.relation, target)
            edge = edges_by_triple.setdefault(
                triple,
                {"source": source, "relation": relation.relation, "target": target, "evidence": []},
            )
            if all(e["chunk_id"] != chunk["chunk_id"] for e in edge["evidence"]):
                edge["evidence"].append(
                    {
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["chunk_id"],
                        "quote": relation.quote,
                    }
                )

    edges = []
    for number, edge in enumerate(edges_by_triple.values(), start=1):
        edges.append({"id": f"e{number}", **edge})

    return {
        "nodes": nodes,
        "edges": edges,
        "merges": merges,
        "possible_missed_merges": possible_missed,
    }


# ---------- build ----------


def load_chunks(corpus: str) -> list[dict]:
    stored = get_collection(corpus).get(include=["documents", "metadatas"])
    chunks = [
        {
            "chunk_id": chunk_id,
            "document_id": (metadata or {}).get("document_id", ""),
            "chunk_index": (metadata or {}).get("chunk_index", 0),
            "chunk_size": (metadata or {}).get("chunk_size"),
            "overlap_percent": (metadata or {}).get("overlap_percent"),
            "text": text or "",
        }
        for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
    ]
    return sorted(chunks, key=lambda c: (c["document_id"], c["chunk_index"]))


def build_graph(corpus: str, limit: int | None = None) -> dict:
    chunks = load_chunks(corpus)
    if not chunks:
        raise RuntimeError(f"Corpus '{corpus}' has no indexed chunks. Index it first.")
    if limit:
        chunks = chunks[:limit]

    extractor = Extractor(corpus)
    rejected, examples, failed = Counter(), [], []
    extractions = []
    started = time.perf_counter()

    for number, chunk in enumerate(chunks, start=1):
        print(f"[{number}/{len(chunks)}] {chunk['document_id']} #{chunk['chunk_index']}")
        data, status = extractor.extract(chunk)

        if data is None:
            failed.append(chunk["chunk_id"])
            continue

        entities, relations = validate_chunk_output(data, chunk, rejected, examples)
        extractions.append((chunk, entities, relations))

    graph = resolve_and_assemble(extractions)
    stats = extractor.stats

    graph["schema"] = {
        "entity_types": list(ENTITY_TYPES),
        "relations": {
            name: {"source": sorted(src), "target": sorted(tgt), "meaning": meaning}
            for name, (src, tgt, meaning) in RELATION_SIGNATURES.items()
        },
    }
    graph["metadata"] = {
        "corpus": corpus,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": GRAPH_MODEL,
        "prompt_version": PROMPT_VERSION,
        "chunk_size": chunks[0]["chunk_size"],
        "overlap_percent": chunks[0]["overlap_percent"],
        "chunks": len(chunks),
        "counts": {
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "merges": len(graph["merges"]),
            "possible_missed_merges": len(graph["possible_missed_merges"]),
        },
        "cost": {
            "llm_calls_this_run": stats["llm_calls"],
            "chunks_from_cache": stats["cached_chunks"],
            "retries": stats["retries"],
            "rate_limited": stats["rate_limited"],
            "prompt_tokens_this_run": stats["prompt_tokens"],
            "completion_tokens_this_run": stats["completion_tokens"],
            "seconds": round(time.perf_counter() - started, 1),
        },
        "failed_chunks": failed,
        "rejected": dict(rejected),
    }
    graph["rejected_examples"] = examples

    # Total cost of the graph, all runs together, from the cache.
    total = Counter()
    for entry in extractor.cache.values():
        total["calls"] += 1
        total["prompt_tokens"] += entry.get("usage", {}).get("prompt_tokens", 0)
        total["completion_tokens"] += entry.get("usage", {}).get("completion_tokens", 0)
        total["llm_seconds"] += entry.get("usage", {}).get("seconds", 0)
    graph["metadata"]["cost"]["total_from_cache"] = {
        k: round(v, 1) if isinstance(v, float) else v for k, v in total.items()
    }

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    output = GRAPH_DIR / f"{corpus}-graph.json"
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {output.relative_to(PROJECT_ROOT)}")
    print(json.dumps(graph["metadata"]["counts"], indent=2))
    print(json.dumps(graph["metadata"]["cost"], indent=2))
    print("rejected:", dict(rejected), "| failed chunks:", len(failed))
    return graph


def main():
    parser = argparse.ArgumentParser(description="Build the concept graph of a corpus.")
    parser.add_argument("--corpus", default="kestrel", choices=["kestrel"])
    parser.add_argument("--limit", type=int, help="Only the first N chunks (for a cheap test).")
    args = parser.parse_args()
    build_graph(args.corpus, limit=args.limit)


if __name__ == "__main__":
    main()
