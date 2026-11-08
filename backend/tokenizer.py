"""Shared tokenizer for the Words tab and BM25.

Both features must see exactly the same words, so they both call tokenize().
"""

from __future__ import annotations

import re

# A word is letters/digits, optionally joined by ' or -  (keeps "kst-5031", "don't").
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*\b")

WORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    """Return the words of a text: lowercased, without stopwords or 1-letter words.

    Order and repetitions are kept, so callers can count occurrences.
    """
    words = []

    for raw_word in TOKEN_PATTERN.findall(text or ""):
        word = raw_word.casefold()

        if len(word) < 2 or word in WORD_STOPWORDS:
            continue

        words.append(word)

    return words
