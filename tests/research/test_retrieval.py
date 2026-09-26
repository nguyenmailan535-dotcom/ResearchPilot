from __future__ import annotations

from pathlib import Path

import numpy as np

from nanobot.research.benchmark import benchmark_retrieval, validate_cases
from nanobot.research.retrieval import DenseRetriever, RetrievalSuite, reflect_query
from nanobot.research.store import ResearchStore


class _KeywordEmbedder:
    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()
        return np.asarray(
            [
                float("rabbitmq" in lowered or "acknowledgement" in lowered),
                float("redis" in lowered or "consumer group" in lowered),
            ],
            dtype=np.float32,
        )

    def passage_embed(self, texts):
        yield from (self._vector(text) for text in texts)

    def query_embed(self, query):
        values = [query] if isinstance(query, str) else query
        yield from (self._vector(text) for text in values)


def _store(tmp_path: Path) -> ResearchStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "rabbit.md").write_text(
        "RabbitMQ uses consumer acknowledgement after message processing." * 5,
        encoding="utf-8",
    )
    (workspace / "redis.md").write_text(
        "Redis Streams tracks delivery with a consumer group and pending entries." * 5,
        encoding="utf-8",
    )
    store = ResearchStore(workspace)
    store.ingest_path(workspace)
    return store


def test_dense_retrieval_builds_cache_and_returns_expected_chunk(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dense = DenseRetriever(store, model_name="test/model", embedder=_KeywordEmbedder())

    index = dense.build_index()
    results = dense.search("RabbitMQ acknowledgement", top_k=1)

    assert index["status"] == "built"
    assert index["chunks"] == 2
    assert results[0]["title"] == "rabbit"
    cached = DenseRetriever(store, model_name="test/model", embedder=_KeywordEmbedder())
    assert cached.build_index()["status"] == "cached"


def test_reflection_expands_cross_language_retrieval_terms() -> None:
    expanded = reflect_query("比较分布式合并和内存开销", [{"title": "ULL"}])

    assert "mergeability" in expanded
    assert "memory space" in expanded
    assert "ULL" not in expanded


def test_benchmark_compares_all_methods(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rabbit = store.search("RabbitMQ acknowledgement", top_k=1)[0]["citation"]
    redis = store.search("Redis consumer group", top_k=1)[0]["citation"]
    cases = [
        {
            "id": "rabbit",
            "query": "RabbitMQ acknowledgement",
            "relevant_citations": [rabbit],
            "language": "en",
            "difficulty": "easy",
            "source": "rabbit",
        },
        {
            "id": "redis",
            "query": "Redis consumer group",
            "relevant_citations": [redis],
            "language": "en",
            "difficulty": "easy",
            "source": "redis",
        },
    ]
    dense = DenseRetriever(store, model_name="test/model", embedder=_KeywordEmbedder())
    dense.build_index()

    result = benchmark_retrieval(store, RetrievalSuite(store, dense), cases, ks=(1, 2))

    assert set(result["methods"]) == {
        "bm25",
        "vector",
        "hybrid_rrf",
        "reflected_hybrid",
    }
    assert result["methods"]["vector"]["metrics"]["at_1"]["mrr"] == 1.0
    assert validate_cases(store, cases) == cases
