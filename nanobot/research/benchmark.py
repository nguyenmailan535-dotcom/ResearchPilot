"""Reproducible multi-method retrieval benchmark for ResearchFlow."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Iterable

from nanobot.research.metrics import percentile
from nanobot.research.retrieval import RetrievalSuite, reflect_query
from nanobot.research.store import ResearchStore

SearchMethod = Callable[[str], list[dict[str, Any]]]


def validate_cases(store: ResearchStore, cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate IDs and labels before running a potentially expensive benchmark."""
    materialized = list(cases)
    ids = [str(case.get("id", "")) for case in materialized]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate evaluation case IDs: {', '.join(duplicates)}")
    labels = [
        str(citation).upper()
        for case in materialized
        for citation in case["relevant_citations"]
    ]
    missing = store.missing_citations(labels)
    if missing:
        raise ValueError(f"Evaluation labels are absent from the corpus: {', '.join(missing)}")
    return materialized


def _aggregate(details: list[dict[str, Any]], k: int) -> dict[str, Any]:
    if not details:
        return {"cases": 0, "recall_at_k": 0.0, "success_at_k": 0.0, "mrr": 0.0}
    recall = 0.0
    success = 0.0
    reciprocal_rank = 0.0
    for detail in details:
        relevant = set(detail["relevant"])
        retrieved = detail["retrieved"][:k]
        hits = relevant.intersection(retrieved)
        recall += len(hits) / len(relevant) if relevant else float(not retrieved)
        success += float(bool(hits))
        rank = next(
            (position for position, citation in enumerate(retrieved, start=1) if citation in relevant),
            None,
        )
        reciprocal_rank += 1.0 / rank if rank else 0.0
    count = len(details)
    return {
        "cases": count,
        "recall_at_k": round(recall / count, 4),
        "success_at_k": round(success / count, 4),
        "mrr": round(reciprocal_rank / count, 4),
    }


def _group_metrics(details: list[dict[str, Any]], *, k: int, field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail.get(field) or "unspecified")].append(detail)
    return {name: _aggregate(rows, k) for name, rows in sorted(grouped.items())}


def benchmark_retrieval(
    store: ResearchStore,
    suite: RetrievalSuite,
    cases: Iterable[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    include_reranker: bool = True,
) -> dict[str, Any]:
    """Evaluate four retrieval methods against the same immutable relevance labels."""
    cases = validate_cases(store, cases)
    ks = tuple(sorted({max(1, int(k)) for k in ks}))
    max_k = max(ks)
    methods: dict[str, SearchMethod] = {
        "bm25": lambda query: suite.bm25(query, top_k=max_k),
        "vector": lambda query: suite.vector(query, top_k=max_k),
        "hybrid_rrf": lambda query: suite.hybrid(query, top_k=max_k),
        "reflected_hybrid": lambda query: suite.reflected_hybrid(query, top_k=max_k),
    }
    if include_reranker and suite.reranker is not None:
        methods["hybrid_reranked"] = lambda query: suite.search(
            query,
            strategy="hybrid",
            top_k=max_k,
            no_answer_threshold=0.0,
            rerank=True,
        ).results
    output: dict[str, Any] = {
        "dataset_cases": len(cases),
        "ks": list(ks),
        "methods": {},
    }
    for method_name, search in methods.items():
        started = time.perf_counter()
        details: list[dict[str, Any]] = []
        for case in cases:
            query = str(case["query"])
            query_started = time.perf_counter()
            results = search(query)
            query_latency_ms = (time.perf_counter() - query_started) * 1000
            detail = {
                "id": str(case["id"]),
                "query": query,
                "language": case.get("language", "unspecified"),
                "difficulty": case.get("difficulty", "unspecified"),
                "source": case.get("source", "unspecified"),
                "relevant": sorted(
                    {str(value).upper() for value in case["relevant_citations"]}
                ),
                "retrieved": [str(item["citation"]).upper() for item in results],
                "latency_ms": round(query_latency_ms, 3),
            }
            if method_name == "reflected_hybrid":
                # Reflection is currently deterministic terminology expansion and
                # does not consume retrieval feedback. Avoid performing a third
                # hybrid search solely to record the expanded query.
                detail["reflected_query"] = reflect_query(query, [])
            details.append(detail)
        metrics = {f"at_{k}": _aggregate(details, k) for k in ks}
        latencies = [float(detail["latency_ms"]) for detail in details]
        misses = [
            detail["id"]
            for detail in details
            if set(detail["relevant"]).isdisjoint(detail["retrieved"][:max_k])
        ]
        output["methods"][method_name] = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "metrics": metrics,
            "by_language_at_max_k": _group_metrics(details, k=max_k, field="language"),
            "by_difficulty_at_max_k": _group_metrics(details, k=max_k, field="difficulty"),
            "details": details,
            "bad_cases": {"retrieval_miss": misses},
        }
    return output
