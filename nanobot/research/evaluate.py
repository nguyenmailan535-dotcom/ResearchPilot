"""Reproducible offline retrieval evaluation for ResearchFlow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from nanobot.research.store import ResearchStore


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL cases containing query and relevant_citations fields."""
    cases: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                case = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            query = case.get("query")
            relevant = case.get("relevant_citations")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Line {line_number}: query must be a non-empty string")
            if not isinstance(relevant, list) or not relevant:
                raise ValueError(f"Line {line_number}: relevant_citations must be a non-empty list")
            normalized = dict(case)
            normalized.update(
                {
                    "id": str(case.get("id") or line_number),
                    "query": query.strip(),
                    "relevant_citations": [str(value).upper() for value in relevant],
                }
            )
            cases.append(normalized)
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")
    return cases


def evaluate_retrieval(
    store: ResearchStore,
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute macro Recall@K, Success@K, and mean reciprocal rank."""
    details: list[dict[str, Any]] = []
    recall_sum = 0.0
    success_sum = 0.0
    reciprocal_rank_sum = 0.0

    for case in cases:
        relevant = {str(value).upper() for value in case["relevant_citations"]}
        results = store.search(str(case["query"]), top_k=top_k)
        retrieved = [item["citation"].upper() for item in results]
        hits = relevant.intersection(retrieved)
        recall = len(hits) / len(relevant)
        first_rank = next(
            (rank for rank, citation in enumerate(retrieved, start=1) if citation in relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        recall_sum += recall
        success_sum += float(bool(hits))
        reciprocal_rank_sum += reciprocal_rank
        details.append(
            {
                "id": str(case.get("id", "")),
                "query": case["query"],
                "relevant": sorted(relevant),
                "retrieved": retrieved,
                "hits": sorted(hits),
                "recall": round(recall, 4),
                "reciprocal_rank": round(reciprocal_rank, 4),
            }
        )

    count = len(details)
    return {
        "cases": count,
        "top_k": top_k,
        "recall_at_k": round(recall_sum / count, 4),
        "success_at_k": round(success_sum / count, 4),
        "mrr": round(reciprocal_rank_sum / count, 4),
        "details": details,
    }
