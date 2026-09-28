"""Optional RAGAS evaluation over saved ResearchFlow end-to-end results."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from nanobot.research.store import ResearchStore

_CITATION = re.compile(r"\[(RF-[0-9a-f]{8}-\d+)\]", re.IGNORECASE)


def _score_value(result: Any) -> float:
    value = getattr(result, "value", result)
    return round(float(value), 4)


def _safe_error(error: BaseException) -> str:
    """Keep actionable provider failures without persisting request identifiers."""
    message = str(error)
    if "402" in message and "Insufficient Balance" in message:
        return f"{type(error).__name__}: provider returned 402 Insufficient Balance"
    message = re.sub(r"request_id[^,})]*", "request_id=<redacted>", message)
    return f"{type(error).__name__}: {message[:500]}"


def aggregate_ragas_details(results: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    """Aggregate completed and partially failed RAGAS case results."""
    count = len(results)
    faithfulness_values = [
        item["faithfulness"] for item in results if item.get("faithfulness") is not None
    ]
    relevancy_values = [
        item["answer_relevancy"]
        for item in results
        if item.get("answer_relevancy") is not None
    ]
    return {
        "model": model,
        "cases": count,
        "completed_cases": sum(bool(item.get("completed")) for item in results),
        "failed_cases": sum(not bool(item.get("completed")) for item in results),
        "faithfulness": round(
            sum(faithfulness_values) / len(faithfulness_values), 4
        ) if faithfulness_values else None,
        "answer_relevancy": round(
            sum(relevancy_values) / len(relevancy_values), 4
        ) if relevancy_values else None,
        "details": results,
    }


def build_ragas_metrics(*, model: str, api_key: str, base_url: str | None) -> tuple[Any, Any]:
    """Create modern RAGAS collection metrics using DeepSeek and local BGE-M3."""
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires the isolated requirements-ragas.txt environment"
        ) from exc
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("RAGAS_REQUEST_TIMEOUT_SECONDS", "180")),
        "max_retries": int(os.getenv("RAGAS_MAX_RETRIES", "1")),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    llm_kwargs: dict[str, Any] = {
        "temperature": 0,
        # Long Chinese research answers can yield many atomic statements.  The
        # RAGAS default is too small and may terminate Instructor JSON with
        # finish_reason=length.
        "max_tokens": int(os.getenv("RAGAS_MAX_TOKENS", "8192")),
    }
    if "deepseek" in model.lower() or "deepseek.com" in (base_url or "").lower():
        # V4 enables high-effort thinking by default. RAGAS uses JSON mode, where
        # reasoning consumes the output budget and can leave Instructor with an
        # incomplete JSON payload. Evaluation needs deterministic non-thinking output.
        llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    llm = llm_factory(
        model,
        client=AsyncOpenAI(**client_kwargs),
        **llm_kwargs,
    )
    embeddings = HuggingFaceEmbeddings(
        model=os.getenv("RESEARCH_EMBEDDING_MODEL", "BAAI/bge-m3"),
        device=os.getenv("RESEARCH_BGE_DEVICE", "cpu"),
    )
    return Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)


async def evaluate_ragas_results(
    store: ResearchStore,
    details: list[dict[str, Any]],
    *,
    faithfulness_metric: Any | None = None,
    relevancy_metric: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    concurrency: int = 2,
    previous_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score metrics against cited chunks, reusing successful per-metric results."""
    resolved_model = model or os.getenv("RAGAS_MODEL", "deepseek-v4-pro")
    if faithfulness_metric is None or relevancy_metric is None:
        resolved_key = api_key or os.getenv("RAGAS_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError("RAGAS_API_KEY or OPENAI_API_KEY is required")
        faithfulness_metric, relevancy_metric = build_ragas_metrics(
            model=resolved_model,
            api_key=resolved_key,
            base_url=base_url or os.getenv("RAGAS_BASE_URL") or None,
        )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def score(detail: dict[str, Any]) -> dict[str, Any]:
        detail_id = str(detail.get("id", ""))
        previous = (previous_details or {}).get(detail_id, {})
        query = str(detail.get("query", ""))
        answer = str(detail.get("answer", ""))
        citations = list(dict.fromkeys(value.upper() for value in _CITATION.findall(answer)))
        contexts = [str(item["content"]) for item in store.get_chunks(citations)]
        result: dict[str, Any] = {
            "id": detail_id,
            "contexts": len(contexts),
        }
        metric_calls = {
            "faithfulness": lambda: faithfulness_metric.ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
            "answer_relevancy": lambda: relevancy_metric.ascore(
                user_input=query, response=answer
            ),
        }
        pending_names: list[str] = []
        pending_calls: list[Any] = []
        for name, call in metric_calls.items():
            if previous.get(name) is not None:
                result[name] = round(float(previous[name]), 4)
            else:
                pending_names.append(name)
                pending_calls.append(call())
        scores: list[Any] = []
        if pending_calls:
            async with semaphore:
                scores = list(
                    await asyncio.gather(*pending_calls, return_exceptions=True)
                )
        errors: dict[str, str] = {}
        for name, value in zip(pending_names, scores, strict=True):
            if isinstance(value, BaseException):
                errors[name] = _safe_error(value)
                result[name] = None
            else:
                result[name] = _score_value(value)
        result["completed"] = not errors
        if errors:
            result["errors"] = errors
        return result

    results = await asyncio.gather(*(score(detail) for detail in details))
    return aggregate_ragas_details(results, model=resolved_model)
