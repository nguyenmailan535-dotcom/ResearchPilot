"""End-to-end answer evaluation for grounded ResearchFlow tasks."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from nanobot.research.metrics import classify_bad_case, estimate_cost_usd, percentile
from nanobot.research.store import ResearchStore
from nanobot.research.verification import verify_claim_evidence

_CITATION = re.compile(r"\[(RF-[0-9a-f]{8}-\d+)\]", re.IGNORECASE)
_ERROR_MARKERS = (
    "sorry, i encountered an error",
    "maximum number of tool call iterations",
    "internal server error",
)
_ABSTENTION_MARKERS = (
    "insufficient_evidence",
    "insufficient evidence",
    "缺少足够证据",
    "没有足够证据",
    "无法从当前论文库",
)


def load_end_to_end_cases(path: Path) -> list[dict[str, Any]]:
    """Load JSONL cases with queries, concept aliases, and evidence labels."""
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
            concepts = case.get("required_concepts")
            citations = case.get("relevant_citations")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Line {line_number}: query must be a non-empty string")
            answerable = bool(case.get("answerable", True))
            if not isinstance(concepts, list) or (answerable and not concepts):
                raise ValueError(
                    f"Line {line_number}: required_concepts must be a list and non-empty "
                    "for answerable cases"
                )
            normalized_concepts: list[list[str]] = []
            for concept in concepts:
                aliases = concept if isinstance(concept, list) else [concept]
                aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
                if not aliases:
                    raise ValueError(f"Line {line_number}: concept aliases cannot be empty")
                normalized_concepts.append(aliases)
            if not isinstance(citations, list) or (answerable and not citations):
                raise ValueError(
                    f"Line {line_number}: relevant_citations must be a list and non-empty "
                    "for answerable cases"
                )
            cases.append(
                {
                    **case,
                    "id": str(case.get("id") or line_number),
                    "query": query.strip(),
                    "required_concepts": normalized_concepts,
                    "relevant_citations": [str(value).upper() for value in citations],
                    "answerable": answerable,
                }
            )
    if not cases:
        raise ValueError("End-to-end evaluation dataset contains no cases")
    return cases


def score_answer(store: ResearchStore, case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Score deterministic answer properties without using an LLM judge."""
    answer = answer or ""
    lowered = answer.casefold()
    cited = list(dict.fromkeys(match.upper() for match in _CITATION.findall(answer)))
    missing = store.missing_citations(cited)
    valid = [citation for citation in cited if citation not in set(missing)]
    relevant = {str(value).upper() for value in case["relevant_citations"]}
    relevant_hits = relevant.intersection(valid)

    concepts: list[dict[str, Any]] = []
    for aliases in case["required_concepts"]:
        matched = next((alias for alias in aliases if alias.casefold() in lowered), None)
        concepts.append({"aliases": aliases, "matched": matched})
    concept_coverage = (
        sum(bool(item["matched"]) for item in concepts) / len(concepts) if concepts else 1.0
    )
    verification = store.verify_report(answer)
    response_ok = bool(answer.strip()) and not any(marker in lowered for marker in _ERROR_MARKERS)
    citation_validity = len(valid) / len(cited) if cited else 0.0
    citation_recall = len(relevant_hits) / len(relevant) if relevant else 1.0
    label_precision = len(relevant_hits) / len(valid) if valid else 0.0
    expected_answerable = bool(case.get("answerable", True))
    abstained = any(marker in lowered for marker in _ABSTENTION_MARKERS)
    completed = response_ok and (
        (bool(valid) and not missing) if expected_answerable else abstained
    )
    answer_pass = (
        completed and concept_coverage >= 0.5 and citation_recall > 0
        if expected_answerable
        else completed
    )
    entailment = verify_claim_evidence(store, answer)

    return {
        "response_ok": response_ok,
        "expected_answerable": expected_answerable,
        "abstained": abstained,
        "completed": completed,
        "answer_pass": answer_pass,
        "concept_coverage": round(concept_coverage, 4),
        "concepts": concepts,
        "citations": cited,
        "valid_citations": valid,
        "missing_citations": missing,
        "citation_validity": round(citation_validity, 4),
        "relevant_citation_recall": round(citation_recall, 4),
        "relevant_label_precision": round(label_precision, 4),
        "relevant_hits": sorted(relevant_hits),
        "paragraph_citation_coverage": verification["citation_coverage"],
        "entailment": entailment,
    }


async def evaluate_agent_answers(
    agent_loop: Any,
    store: ResearchStore,
    cases: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    judge: Any | None = None,
    claim_judge: Any | None = None,
    prompt_usd_per_million: float = 0.0,
    completion_usd_per_million: float = 0.0,
) -> dict[str, Any]:
    """Run independent Agent sessions and aggregate answer-grounding metrics."""
    selected = list(cases)
    if limit is not None:
        selected = selected[: max(1, limit)]
    missing = store.missing_citations(
        citation for case in selected for citation in case["relevant_citations"]
    )
    if missing:
        raise ValueError(f"Evaluation labels are absent from the corpus: {', '.join(missing)}")

    details: list[dict[str, Any]] = []
    for case in selected:
        started = time.perf_counter()
        progress_events: list[str] = []
        streamed_chars = 0

        async def on_progress(content: str, *, tool_hint: bool = False) -> None:
            progress_events.append(content)

        async def on_stream(content: str) -> None:
            nonlocal streamed_chars
            streamed_chars += len(content)

        prompt = (
            "Use the research-flow skill. Answer the question using only the indexed corpus. "
            "Give a concise direct answer, include exact quantitative details where applicable, "
            "and cite every factual paragraph with RF citations. Do not create a report file.\n\n"
            f"Question: {case['query']}"
        )
        error = None
        try:
            response = await agent_loop.process_direct(
                content=prompt,
                session_key=f"e2e:{case['id']}:{uuid.uuid4().hex[:8]}",
                channel="evaluation",
                chat_id=str(case["id"]),
                on_progress=on_progress,
                on_stream=on_stream,
            )
            answer = "" if response is None else str(getattr(response, "content", response) or "")
        except Exception as exc:
            answer = ""
            error = str(exc)
        elapsed = time.perf_counter() - started
        usage = dict(getattr(agent_loop, "_last_usage", {}) or {})
        score = score_answer(store, case, answer)
        detail = {
                "id": case["id"],
                "query": case["query"],
                "language": case.get("language", "unspecified"),
                "difficulty": case.get("difficulty", "unspecified"),
                "answer": answer,
                "error": error,
                "elapsed_seconds": round(elapsed, 3),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "tool_events": progress_events,
                "streamed_chars": streamed_chars,
                **score,
            }
        pricing_configured = bool(prompt_usd_per_million or completion_usd_per_million)
        detail["estimated_cost_usd"] = (
            estimate_cost_usd(
                detail["prompt_tokens"],
                detail["completion_tokens"],
                prompt_usd_per_million=prompt_usd_per_million,
                completion_usd_per_million=completion_usd_per_million,
            )
            if pricing_configured
            else None
        )
        if judge is not None:
            try:
                detail["llm_judge"] = await judge(case, answer)
            except Exception as exc:
                detail["llm_judge"] = {"error": str(exc)}
        if claim_judge is not None:
            try:
                detail["semantic_entailment"] = await claim_judge(answer)
            except Exception as exc:
                detail["semantic_entailment"] = {"error": str(exc)}
        detail["bad_case_labels"] = classify_bad_case(detail)
        details.append(detail)

    count = len(details)

    def mean(field: str) -> float:
        return round(sum(float(item[field]) for item in details) / max(1, count), 4)

    latencies = [float(item["elapsed_seconds"]) for item in details]
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in details)
    completion_tokens = sum(int(item["completion_tokens"]) for item in details)
    answerable = [item for item in details if item["expected_answerable"]]
    unanswerable = [item for item in details if not item["expected_answerable"]]
    bad_case_counts: dict[str, int] = {}
    for item in details:
        for label in item["bad_case_labels"]:
            bad_case_counts[label] = bad_case_counts.get(label, 0) + 1
    judge_scores = [
        float(item["llm_judge"]["overall"])
        for item in details
        if "llm_judge" in item and "overall" in item["llm_judge"]
    ]
    semantic_support = [
        float(item["semantic_entailment"]["support_rate"])
        for item in details
        if "semantic_entailment" in item and "support_rate" in item["semantic_entailment"]
    ]
    metrics = {
            "task_completion_rate": mean("completed"),
            "answer_pass_rate": mean("answer_pass"),
            "concept_coverage": mean("concept_coverage"),
            "citation_validity": mean("citation_validity"),
            "relevant_citation_recall": mean("relevant_citation_recall"),
            "paragraph_citation_coverage": mean("paragraph_citation_coverage"),
            "claim_support_rate": round(
                sum(float(item["entailment"]["support_rate"]) for item in details)
                / max(1, count),
                4,
            ),
            "average_latency_seconds": mean("elapsed_seconds"),
            "p50_latency_seconds": percentile(latencies, 50),
            "p95_latency_seconds": percentile(latencies, 95),
            "average_prompt_tokens": mean("prompt_tokens"),
            "average_completion_tokens": mean("completion_tokens"),
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "estimated_cost_usd": round(
                sum(float(item["estimated_cost_usd"]) for item in details), 8
            ) if pricing_configured else None,
            "average_tool_events": round(
                sum(len(item["tool_events"]) for item in details) / max(1, count), 4
            ),
            "unanswerable_recall": round(
                sum(bool(item["abstained"]) for item in unanswerable) / len(unanswerable), 4
            ) if unanswerable else None,
            "false_answer_rate": round(
                sum(not bool(item["abstained"]) for item in unanswerable) / len(unanswerable), 4
            ) if unanswerable else None,
            "answerable_accuracy": round(
                sum(bool(item["answer_pass"]) for item in answerable) / len(answerable), 4
            ) if answerable else None,
            "llm_judge_overall": round(sum(judge_scores) / len(judge_scores), 4)
            if judge_scores else None,
            "semantic_claim_support_rate": round(
                sum(semantic_support) / len(semantic_support), 4
            ) if semantic_support else None,
        }
    return {
        "cases": count,
        "metrics": metrics,
        "bad_case_summary": dict(sorted(bad_case_counts.items())),
        "pricing": {
            "configured": pricing_configured,
            "prompt_usd_per_million": prompt_usd_per_million,
            "completion_usd_per_million": completion_usd_per_million,
        },
        "details": details,
    }
