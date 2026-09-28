from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nanobot.research.benchmark import benchmark_retrieval, tune_rrf_weights, validate_cases
from nanobot.research.retrieval import (
    BGEM3DenseEmbedder,
    DenseRetriever,
    RetrievalSuite,
    reflect_query,
)
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


def test_bge_model_initialization_is_singleton_under_concurrent_queries(monkeypatch) -> None:
    instances: list[object] = []

    class FakeModel:
        def __init__(self, *_args, **_kwargs) -> None:
            time.sleep(0.05)
            instances.append(self)

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(BGEM3FlagModel=FakeModel),
    )
    embedder = BGEM3DenseEmbedder()
    with ThreadPoolExecutor(max_workers=4) as executor:
        models = list(executor.map(lambda _index: embedder.model, range(4)))

    assert len(instances) == 1
    assert all(model is instances[0] for model in models)


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


def test_rrf_tuning_reuses_retrieval_and_prefers_recall(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rabbit = store.search("RabbitMQ acknowledgement", top_k=1)[0]["citation"]
    redis = store.search("Redis consumer group", top_k=1)[0]["citation"]
    cases = [
        {
            "id": "rabbit",
            "query": "RabbitMQ acknowledgement",
            "relevant_citations": [rabbit],
        },
        {
            "id": "redis",
            "query": "Redis consumer group",
            "relevant_citations": [redis],
        },
    ]
    dense = DenseRetriever(store, model_name="test/model", embedder=_KeywordEmbedder())
    dense.build_index()
    result = tune_rrf_weights(
        store,
        RetrievalSuite(store, dense),
        cases,
        top_k=2,
        weight_ratios=(0.5, 1.0, 2.0),
        rank_constants=(20,),
    )

    assert result["best"] is not None
    assert result["best"]["metrics"]["recall_at_k"] >= 0.0
    assert len(result["candidates"]) == 3


def test_online_hybrid_supports_abstention_fallback_and_reranking(tmp_path: Path) -> None:
    store = _store(tmp_path)
    dense = DenseRetriever(store, model_name="test/model", embedder=_KeywordEmbedder())
    suite = RetrievalSuite(
        store,
        dense,
        reranker=lambda _query, passages: [float("RabbitMQ" in text) for text in passages],
    )

    outcome = suite.search(
        "RabbitMQ acknowledgement",
        strategy="hybrid",
        top_k=1,
        no_answer_threshold=0.0,
        rerank=True,
    )
    assert outcome.used_strategy == "hybrid"
    assert outcome.reranked is True
    assert outcome.results[0]["title"] == "rabbit"
    assert outcome.latency_ms >= 0

    abstained = suite.search(
        "totally absent vocabulary",
        strategy="bm25",
        top_k=2,
        no_answer_threshold=0.1,
    )
    assert abstained.no_answer is True
    assert abstained.results == []

    class BrokenDense:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("embedding model offline")

    degraded = RetrievalSuite(store, BrokenDense()).search(
        "RabbitMQ acknowledgement",
        strategy="hybrid",
        no_answer_threshold=0.0,
    )
    assert degraded.degraded is True
    assert degraded.used_strategy == "bm25"

    class BrokenBackend:
        def bm25(self, *_args, **_kwargs):
            raise OSError("Milvus offline")

        def vector(self, *_args, **_kwargs):
            raise OSError("Milvus offline")

    external_degraded = RetrievalSuite(store, dense, backend=BrokenBackend()).search(
        "RabbitMQ acknowledgement",
        strategy="hybrid",
        no_answer_threshold=0.0,
    )
    assert external_degraded.degraded is True
    assert external_degraded.used_strategy == "bm25"
    assert external_degraded.results[0]["title"] == "rabbit"
    assert "sparse backend unavailable" in external_degraded.fallback_reason
