"""BM25 lexical retrieval: an inverted index plus one scoring formula.

The index is built from the chunks already stored in Chroma, so BM25 and dense
retrieval always rank exactly the same chunks.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path

from backend.tokenizer import tokenize

INDEX_FORMAT_VERSION = 1


class BM25Index:
    """Okapi BM25 over a list of chunks.

    Each chunk is a dict with "chunk_id", "document_id" and "text".
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # how fast repeated words stop adding score
        self.b = b  # how strongly long chunks are penalised

        # Inverted index: word -> {chunk_id: times the word appears in that chunk}
        self.postings: dict[str, dict[str, int]] = {}
        # Chunk length in tokens, and which document each chunk belongs to.
        self.lengths: dict[str, int] = {}
        self.document_ids: dict[str, str] = {}

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            tokens = tokenize(chunk["text"])

            self.lengths[chunk_id] = len(tokens)
            self.document_ids[chunk_id] = chunk["document_id"]

            for word, count in Counter(tokens).items():
                self.postings.setdefault(word, {})[chunk_id] = count

        self.n_chunks = len(self.lengths)
        self.avgdl = (
            sum(self.lengths.values()) / self.n_chunks if self.n_chunks else 0.0
        )

    def chunk_frequency(self, word: str) -> int:
        """Number of chunks that contain the word."""
        return len(self.postings.get(word, {}))

    def idf(self, word: str) -> float:
        """Rare words weigh more: ln((N - n + 0.5) / (n + 0.5) + 1)."""
        n = self.chunk_frequency(word)
        return math.log((self.n_chunks - n + 0.5) / (n + 0.5) + 1)

    def search(
        self,
        question: str,
        allowed_document_ids: set[str] | None = None,
        n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return up to n (chunk_id, score) pairs, best first. Scores are > 0."""
        if not self.n_chunks:
            return []

        scores: dict[str, float] = {}

        for word in tokenize(question):
            posting = self.postings.get(word)

            if not posting:
                continue  # word absent from the corpus: it cannot change the ranking

            idf = self.idf(word)

            # Only chunks that contain the word are visited (inverted index).
            for chunk_id, frequency in posting.items():
                if (
                    allowed_document_ids is not None
                    and self.document_ids[chunk_id] not in allowed_document_ids
                ):
                    continue

                length_norm = 1 - self.b + self.b * self.lengths[chunk_id] / self.avgdl
                term_score = (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * length_norm)
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        # Sort by score, then by chunk id so ties always come out in the same order.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:n]

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "format_version": INDEX_FORMAT_VERSION,
            "k1": self.k1,
            "b": self.b,
            "n_chunks": self.n_chunks,
            "avgdl": self.avgdl,
            "lengths": self.lengths,
            "document_ids": self.document_ids,
            "postings": self.postings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        if data.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError("Unsupported BM25 index format. Rebuild the index.")

        index = cls.__new__(cls)
        index.k1 = data["k1"]
        index.b = data["b"]
        index.n_chunks = data["n_chunks"]
        index.avgdl = data["avgdl"]
        index.lengths = data["lengths"]
        index.document_ids = data["document_ids"]
        index.postings = data["postings"]
        return index

    def save(self, path: Path) -> None:
        """Write the index as JSON. Write to a temp file first so a crash
        never leaves a half-written index behind."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(temp_path, path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
