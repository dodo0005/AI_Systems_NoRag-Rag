"""Test setup: no network, no API key, no model download.

- sentence_transformers is replaced by a tiny hashing encoder, so Chroma works
  offline and fast (the real model is only needed to run the app);
- the LLM is replaced by a fake that returns canned JSON (tests never pay);
- Chroma, BM25, graphs and traces go to a temporary folder, never to the repo's.
"""

import hashlib
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

TMP = Path(tempfile.mkdtemp(prefix="norag-tests-"))
os.environ["CHROMA_DIR"] = str(TMP / "chroma")
os.environ["BM25_DIR"] = str(TMP / "bm25")
os.environ["TRACE_DB"] = str(TMP / "traces.db")
os.environ["GRAPH_CALL_DELAY"] = "0"
os.environ["LLM_URL"] = "http://llm.invalid/v1"
os.environ["LLM_API_KEY"] = "test-key-123456"
os.environ["LLM_MODEL"] = "fake-model"
os.environ["LLM_PRICE_INPUT_PER_M"] = "1.0"
os.environ["LLM_PRICE_OUTPUT_PER_M"] = "2.0"


class _FakeSentenceTransformer:
    """Bag of hashed words, normalised: similar words give similar vectors."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False, **kwargs):
        vectors = []
        for text in texts:
            v = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            n = np.linalg.norm(v)
            vectors.append(v / n if n else v)
        return np.array(vectors)


_fake = types.ModuleType("sentence_transformers")
_fake.SentenceTransformer = _FakeSentenceTransformer
_fake.__version__ = "0.0-test"
sys.modules.setdefault("sentence_transformers", _fake)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def corpus_dirs(monkeypatch, tmp_path):
    """Point both corpora to empty temporary folders and fresh collections."""
    from backend import rag, graph, graph_build, main

    for name in rag.CORPORA:
        docs = tmp_path / name / "docs"
        docs.mkdir(parents=True)
        monkeypatch.setitem(rag.CORPORA, name, {
            "docs_dir": docs,
            "collection_name": f"test_{name}_{tmp_path.name}"[:60],
        })
    monkeypatch.setattr(graph, "GRAPH_DIR", tmp_path / "graph")
    monkeypatch.setattr(graph_build, "GRAPH_DIR", tmp_path / "graph")
    monkeypatch.setattr(graph_build, "CACHE_DIR", tmp_path / "graph" / "cache")
    monkeypatch.setattr(main, "QUARANTINE_DIR", tmp_path / "quarantine")
    rag._bm25_cache.clear()
    rag._graph_cache.clear()
    return tmp_path


class FakeLLM:
    """Replaces backend.llm.chat. `replies` is a function prompt -> JSON string."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        from backend import llm
        from backend.tracing import current_trace

        self.prompts.append(prompt)
        content = self.reply(prompt)
        result = llm.LLMResult(content=content, input_tokens=100, output_tokens=20,
                               cost_usd=llm.cost_usd(100, 20), model="fake-model", seconds=0.01)
        tr = current_trace()
        if tr is not None:
            tr.add_usage(100, 20, result.cost_usd)
        return result


@pytest.fixture
def fake_llm(monkeypatch):
    def install(reply):
        from backend import llm

        fake = FakeLLM(reply)
        monkeypatch.setattr(llm, "chat", fake)
        return fake

    return install
