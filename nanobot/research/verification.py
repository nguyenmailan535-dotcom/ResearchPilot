"""Claim-to-evidence verification for ResearchFlow reports."""

from __future__ import annotations

import re
from typing import Any

from nanobot.research.store import ResearchStore, tokenize

_CITATION = re.compile(r"\[(RF-[0-9a-f]{8}-\d+)\]", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
_SENTENCE = re.compile(r"(?<=[。！？!?])\s+|(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_GENERIC = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "or", "in",
    "for", "with", "that", "this", "it", "as", "be", "by", "from", "on", "can",
}


def extract_claims(content: str) -> list[dict[str, Any]]:
    """Extract citation-bearing atomic claims while preserving their evidence IDs."""
    claims: list[dict[str, Any]] = []
    for sentence in _SENTENCE.split(content or ""):
        text = sentence.strip().lstrip("-* ")
        citations = list(dict.fromkeys(value.upper() for value in _CITATION.findall(text)))
        plain = _CITATION.sub("", text).strip()
        if citations and len(plain) >= 12:
            claims.append({"claim": plain, "citations": citations})
    return claims


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1 and token not in _GENERIC}


def verify_claim_evidence(store: ResearchStore, content: str) -> dict[str, Any]:
    """Run deterministic lexical, numeric and citation checks before any LLM judge."""
    results: list[dict[str, Any]] = []
    for item in extract_claims(content):
        evidence_rows = store.get_chunks(item["citations"])
        evidence = "\n".join(str(row["content"]) for row in evidence_rows)
        claim_tokens = _content_tokens(item["claim"])
        evidence_tokens = _content_tokens(evidence)
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        claim_numbers = set(_NUMBER.findall(item["claim"]))
        evidence_numbers = set(_NUMBER.findall(evidence))
        missing_numbers = sorted(claim_numbers - evidence_numbers)
        missing_citations = store.missing_citations(item["citations"])
        if missing_citations or not evidence_rows or overlap < 0.05 or missing_numbers:
            label = "NOT_SUPPORTED"
        elif overlap < 0.18:
            label = "PARTIALLY_SUPPORTED"
        else:
            label = "ENTAILED"
        results.append(
            {
                **item,
                "label": label,
                "lexical_overlap": round(overlap, 4),
                "missing_numbers": missing_numbers,
                "missing_citations": missing_citations,
                "evidence_excerpt": evidence[:600],
            }
        )
    counts = {
        label: sum(result["label"] == label for result in results)
        for label in ("ENTAILED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")
    }
    supported = counts["ENTAILED"] + counts["PARTIALLY_SUPPORTED"]
    return {
        "claims": len(results),
        "entailed": counts["ENTAILED"],
        "partially_supported": counts["PARTIALLY_SUPPORTED"],
        "unsupported": counts["NOT_SUPPORTED"],
        "support_rate": round(supported / len(results), 4) if results else 1.0,
        "details": results,
    }
