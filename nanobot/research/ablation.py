"""Chunk-size and structure-aware ingestion ablation for labeled corpora."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from nanobot.research.metrics import percentile
from nanobot.research.retrieval import DenseRetriever, RetrievalSuite
from nanobot.research.store import ResearchStore, tokenize

DEFAULT_CHUNK_CONFIGS = (
    (256, 64, "fixed"),
    (384, 64, "fixed"),
    (480, 80, "fixed"),
    (768, 120, "fixed"),
    (480, 80, "structure"),
)


def _matches_reference(candidate: dict[str, Any], references: list[dict[str, Any]]) -> bool:
    candidate_tokens = set(tokenize(str(candidate["content"])))
    for reference in references:
        if candidate["source_id"] != reference["source_id"]:
            continue
        if reference.get("page") and candidate.get("page") == reference["page"]:
            return True
        reference_tokens = set(tokenize(str(reference["content"])))
        overlap = len(candidate_tokens & reference_tokens) / max(1, len(reference_tokens))
        if overlap >= 0.2:
            return True
    return False


def run_chunk_ablation(
    store: ResearchStore,
    cases: Iterable[dict[str, Any]],
    *,
    model_name: str,
    configs: Iterable[tuple[int, int, str]] = DEFAULT_CHUNK_CONFIGS,
    top_k: int = 5,
    embedder_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Re-index the same files under isolated configs and compare hybrid retrieval."""
    cases = list(cases)
    references = {
        str(case["id"]): store.get_chunks(case["relevant_citations"]) for case in cases
    }
    sources = [Path(item["path"]) for item in store.list_sources()]
    shared_embedder = (
        embedder_factory()
        if embedder_factory
        else DenseRetriever(store, model_name=model_name).embedder
    )
    results: list[dict[str, Any]] = []
    for chunk_size, overlap, strategy in configs:
        with tempfile.TemporaryDirectory(prefix="researchflow-ablation-") as temporary:
            variant = ResearchStore(Path(temporary))
            ingest_started = time.perf_counter()
            for source in sources:
                variant.ingest_file(
                    source,
                    max_chars=chunk_size,
                    overlap=overlap,
                    chunk_strategy=strategy,
                )
            ingest_seconds = time.perf_counter() - ingest_started
            dense = DenseRetriever(
                variant,
                model_name=model_name,
                embedder=shared_embedder,
                segment_chars=chunk_size,
                segment_overlap=overlap,
            )
            index = dense.build_index()
            suite = RetrievalSuite(variant, dense)
            reciprocal_ranks: list[float] = []
            recalls: list[float] = []
            latencies: list[float] = []
            bad_cases: list[str] = []
            for case in cases:
                started = time.perf_counter()
                retrieved = suite.hybrid(str(case["query"]), top_k=top_k)
                latencies.append((time.perf_counter() - started) * 1000)
                ranks = [
                    rank
                    for rank, candidate in enumerate(retrieved, start=1)
                    if _matches_reference(candidate, references[str(case["id"])])
                ]
                hit = bool(ranks)
                recalls.append(float(hit))
                reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
                if not hit:
                    bad_cases.append(str(case["id"]))
            results.append(
                {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "strategy": strategy,
                    "chunks": variant.stats()["chunks"],
                    "segments": index["segments"],
                    "ingest_seconds": round(ingest_seconds, 3),
                    "recall_at_k": round(sum(recalls) / max(1, len(recalls)), 4),
                    "mrr": round(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)), 4),
                    "p50_latency_ms": percentile(latencies, 50),
                    "p95_latency_ms": percentile(latencies, 95),
                    "bad_cases": bad_cases,
                }
            )
    ranked = sorted(
        results,
        key=lambda item: (-item["recall_at_k"], -item["mrr"], item["p95_latency_ms"]),
    )
    return {
        "cases": len(cases),
        "top_k": top_k,
        "selection_rule": "max recall_at_k, then max mrr, then min p95_latency_ms",
        "recommended": ranked[0] if ranked else None,
        "results": results,
    }
