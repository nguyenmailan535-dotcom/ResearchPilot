"""Structured, evidence-aware LLM-as-Judge used only by offline evaluation."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from nanobot.providers.base import LLMProvider
from nanobot.research.store import ResearchStore
from nanobot.research.verification import extract_claims

_JUDGE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "score_grounded_answer",
            "description": "Score an answer using only the supplied gold evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "correctness": {"type": "integer", "minimum": 0, "maximum": 2},
                    "faithfulness": {"type": "integer", "minimum": 0, "maximum": 2},
                    "completeness": {"type": "integer", "minimum": 0, "maximum": 2},
                    "citation_alignment": {"type": "integer", "minimum": 0, "maximum": 2},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                    "contradictions": {"type": "array", "items": {"type": "string"}},
                    "evidence_spans": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "correctness", "faithfulness", "completeness", "citation_alignment",
                    "unsupported_claims", "contradictions", "evidence_spans", "reason", "confidence",
                ],
            },
        },
    }
]

_RUBRIC = """You are an independent evaluator. Use only the supplied evidence.
Score each dimension 0, 1, or 2:
- correctness: 0 materially wrong, 1 mixed/minor error, 2 correct.
- faithfulness: 0 unsupported/contradicted, 1 partially supported, 2 fully supported.
- completeness: 0 misses essentials, 1 partially covers them, 2 covers all required concepts.
- citation_alignment: 0 citations do not support claims, 1 mixed, 2 every citation aligns.
Gold evidence defines the expected answer. Cited evidence contains the original chunks referenced
by the answer and must also be used when checking faithfulness and citation alignment.
Call score_grounded_answer exactly once. Do not reward fluent wording or outside knowledge."""

_ENTAILMENT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "verify_claim_entailment",
            "description": "Classify every claim against only its cited evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_index": {"type": "integer", "minimum": 0},
                                "label": {
                                    "type": "string",
                                    "enum": [
                                        "ENTAILED", "PARTIALLY_SUPPORTED",
                                        "NOT_SUPPORTED", "CONTRADICTED",
                                    ],
                                },
                                "evidence_span": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["claim_index", "label", "evidence_span", "reason"],
                        },
                    }
                },
                "required": ["verdicts"],
            },
        },
    }
]

_ENTAILMENT_RUBRIC = """Classify each atomic claim using only its cited evidence.
ENTAILED means the evidence directly supports the full claim; PARTIALLY_SUPPORTED means only part;
NOT_SUPPORTED means the evidence is insufficient; CONTRADICTED means it states the opposite.
Check quantities, entities, scope and comparisons. Return one verdict per claim in index order."""


def _validated_score(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    for field in ("correctness", "faithfulness", "completeness", "citation_alignment"):
        result[field] = max(0, min(2, int(result.get(field, 0))))
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    for field in ("unsupported_claims", "contradictions", "evidence_spans"):
        value = result.get(field, [])
        result[field] = [str(item) for item in value] if isinstance(value, list) else []
    result["reason"] = str(result.get("reason", ""))
    result["overall"] = round(
        sum(result[field] for field in (
            "correctness", "faithfulness", "completeness", "citation_alignment"
        )) / 8,
        4,
    )
    return result


async def judge_answer(
    provider: LLMProvider,
    model: str,
    store: ResearchStore,
    case: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    """Score one answer with a fixed rubric, JSON schema and original gold evidence."""
    gold_citations = [str(value).upper() for value in case.get("relevant_citations", [])]
    evidence = store.get_chunks(gold_citations)
    cited_citations = list(
        dict.fromkeys(
            citation
            for claim in extract_claims(answer)
            for citation in claim["citations"]
            if citation not in gold_citations
        )
    )
    cited_evidence = store.get_chunks(cited_citations)
    payload = {
        "question": case["query"],
        "required_concepts": case.get("required_concepts", []),
        "expected_answerable": case.get("answerable", True),
        "answer": answer,
        "gold_evidence": [
            {"citation": row["citation"], "title": row["title"], "content": row["content"]}
            for row in evidence
        ],
        "answer_cited_evidence": [
            {"citation": row["citation"], "title": row["title"], "content": row["content"]}
            for row in cited_evidence
        ],
    }
    response = await provider.chat_with_retry(
        messages=[
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tools=_JUDGE_TOOL,
        model=model,
        tool_choice={"type": "function", "function": {"name": "score_grounded_answer"}},
        # DeepSeek ignores temperature in thinking mode. A fixed rubric benefits
        # more from repeatability, so use V4 Pro in non-thinking mode.
        max_tokens=2048,
        temperature=0.0,
        reasoning_effort="none",
    )
    if not response.has_tool_calls:
        detail = (response.content or response.finish_reason or "empty response")[:300]
        raise RuntimeError(
            f"LLM judge did not return the required structured tool call: {detail}"
        )
    result = _validated_score(response.tool_calls[0].arguments)
    result["usage"] = dict(response.usage or {})
    return result


async def judge_claim_entailment(
    provider: LLMProvider,
    model: str,
    store: ResearchStore,
    content: str,
) -> dict[str, Any]:
    """Run optional semantic Claim—Evidence verification for offline/asynchronous QA."""
    claims = extract_claims(content)
    payload: list[dict[str, Any]] = []
    for index, item in enumerate(claims):
        payload.append(
            {
                "claim_index": index,
                "claim": item["claim"],
                "evidence": [
                    {"citation": row["citation"], "content": row["content"]}
                    for row in store.get_chunks(item["citations"])
                ],
            }
        )
    if not payload:
        return {"claims": 0, "support_rate": 1.0, "verdicts": []}
    response = await provider.chat_with_retry(
        messages=[
            {"role": "system", "content": _ENTAILMENT_RUBRIC},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tools=_ENTAILMENT_TOOL,
        model=model,
        tool_choice={"type": "function", "function": {"name": "verify_claim_entailment"}},
        max_tokens=4096,
        temperature=0.0,
        reasoning_effort="none",
    )
    if not response.has_tool_calls:
        detail = (response.content or response.finish_reason or "empty response")[:300]
        raise RuntimeError(
            f"Claim entailment judge did not return the required tool call: {detail}"
        )
    verdicts = response.tool_calls[0].arguments.get("verdicts", [])
    if not isinstance(verdicts, list) or len(verdicts) != len(payload):
        raise RuntimeError("Claim entailment judge returned incomplete verdicts")
    supported = sum(
        item.get("label") in {"ENTAILED", "PARTIALLY_SUPPORTED"} for item in verdicts
    )
    return {
        "claims": len(payload),
        "support_rate": round(supported / len(payload), 4),
        "verdicts": verdicts,
        "usage": dict(response.usage or {}),
    }


async def judge_saved_evaluation(
    provider: LLMProvider,
    model: str,
    store: ResearchStore,
    cases: list[dict[str, Any]],
    evaluation: dict[str, Any],
    *,
    limit: int | None = None,
    concurrency: int = 2,
    semantic_entailment: bool = True,
) -> dict[str, Any]:
    """Judge saved answers without regenerating them or mutating the original result."""
    output = copy.deepcopy(evaluation)
    case_by_id = {str(case["id"]): case for case in cases}
    details = output.get("details", [])
    if limit is not None:
        details = details[: max(1, limit)]
    gate = asyncio.Semaphore(max(1, concurrency))

    async def score(detail: dict[str, Any]) -> dict[str, Any]:
        case = case_by_id.get(str(detail.get("id")))
        if case is None:
            raise ValueError(f"Saved evaluation case is absent from dataset: {detail.get('id')}")
        answer = str(detail.get("answer", ""))
        async with gate:
            answer_score = await judge_answer(provider, model, store, case, answer)
            entailment = (
                await judge_claim_entailment(provider, model, store, answer)
                if semantic_entailment
                else None
            )
        enriched = copy.deepcopy(detail)
        enriched["llm_judge"] = answer_score
        if entailment is not None:
            enriched["semantic_entailment"] = entailment
        return enriched

    judged = await asyncio.gather(*(score(detail) for detail in details))
    usage_blocks = [
        block
        for item in judged
        for block in (item.get("llm_judge"), item.get("semantic_entailment"))
        if isinstance(block, dict)
    ]
    prompt_tokens = sum(
        int(block.get("usage", {}).get("prompt_tokens", 0) or 0)
        for block in usage_blocks
    )
    completion_tokens = sum(
        int(block.get("usage", {}).get("completion_tokens", 0) or 0)
        for block in usage_blocks
    )
    output["details"] = judged
    output["judge"] = {
        "model": model,
        "cases": len(judged),
        "concurrency": max(1, concurrency),
        "semantic_entailment_enabled": semantic_entailment,
        "average_overall": round(
            sum(float(item["llm_judge"]["overall"]) for item in judged) / max(1, len(judged)),
            4,
        ),
        "semantic_claim_support_rate": (
            round(
                sum(float(item["semantic_entailment"]["support_rate"]) for item in judged)
                / max(1, len(judged)),
                4,
            )
            if semantic_entailment
            else None
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return output
