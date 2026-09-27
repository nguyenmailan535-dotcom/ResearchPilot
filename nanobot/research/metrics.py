"""Shared observability and bad-case helpers for ResearchFlow evaluations."""

from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a heavy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    percentile_value = max(0.0, min(100.0, percentile_value))
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 4)


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_usd_per_million: float = 0.0,
    completion_usd_per_million: float = 0.0,
) -> float:
    """Estimate cost with explicit, versionable prices instead of hard-coded vendor rates."""
    cost = (
        prompt_tokens * prompt_usd_per_million
        + completion_tokens * completion_usd_per_million
    ) / 1_000_000
    return round(cost, 8)


def classify_bad_case(detail: dict[str, Any]) -> list[str]:
    """Assign actionable regression categories to an end-to-end evaluation row."""
    labels: list[str] = []
    if detail.get("error") or not detail.get("response_ok", True):
        labels.append("runtime_or_provider_failure")
    if detail.get("expected_answerable") is False and not detail.get("abstained", False):
        labels.append("false_answer")
    if detail.get("expected_answerable") is True and detail.get("abstained", False):
        labels.append("false_refusal")
    if detail.get("expected_answerable", True) and not detail.get("relevant_hits"):
        labels.append("retrieval_or_ranking_miss")
    if detail.get("citation_validity", 1.0) < 1.0:
        labels.append("invalid_citation")
    if detail.get("paragraph_citation_coverage", 1.0) < 1.0:
        labels.append("uncited_claim")
    if detail.get("concept_coverage", 1.0) < 0.5:
        labels.append("incomplete_answer")
    if detail.get("entailment", {}).get("unsupported", 0):
        labels.append("unsupported_claim")
    return labels
