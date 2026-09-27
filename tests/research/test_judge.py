from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.research.judge import (
    judge_answer,
    judge_claim_entailment,
    judge_saved_evaluation,
)
from nanobot.research.store import ResearchStore


class _JudgeProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        assert kwargs["temperature"] == 0.0
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["tool_choice"]["function"]["name"] == "score_grounded_answer"
        assert "gold_evidence" in messages[1]["content"]
        assert "answer_cited_evidence" in messages[1]["content"]
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="judge",
                    name="score_grounded_answer",
                    arguments={
                        "correctness": 2,
                        "faithfulness": 2,
                        "completeness": 1,
                        "citation_alignment": 2,
                        "unsupported_claims": [],
                        "contradictions": [],
                        "evidence_spans": ["publisher confirms"],
                        "reason": "Grounded but concise.",
                        "confidence": 0.9,
                    },
                )
            ],
        )

    def get_default_model(self) -> str:
        return "judge"


class _EntailmentProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        assert kwargs["temperature"] == 0.0
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["tool_choice"]["function"]["name"] == "verify_claim_entailment"
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="entailment",
                    name="verify_claim_entailment",
                    arguments={
                        "verdicts": [
                            {
                                "claim_index": 0,
                                "label": "ENTAILED",
                                "evidence_span": "publisher confirms",
                                "reason": "Direct support.",
                            }
                        ]
                    },
                )
            ],
        )

    def get_default_model(self) -> str:
        return "judge"


class _CombinedProvider(_JudgeProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        function_name = tools[0]["function"]["name"]
        if function_name == "verify_claim_entailment":
            return await _EntailmentProvider().chat(messages, tools, model, **kwargs)
        return await super().chat(messages, tools, model, **kwargs)


@pytest.mark.asyncio
async def test_llm_judge_uses_structured_evidence_aware_rubric(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.md"
    source.write_text("RabbitMQ supports publisher confirms." * 5)
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("publisher confirms")[0]["citation"]

    result = await judge_answer(
        _JudgeProvider(),
        "judge-model",
        store,
        {
            "query": "How is delivery confirmed?",
            "required_concepts": [["publisher confirms"]],
            "relevant_citations": [citation],
        },
        f"Use publisher confirms [{citation}].",
    )

    assert result["overall"] == 0.875
    assert result["confidence"] == 0.9

    entailment = await judge_claim_entailment(
        _EntailmentProvider(),
        "judge-model",
        store,
        f"RabbitMQ supports publisher confirms [{citation}].",
    )
    assert entailment["support_rate"] == 1.0
    assert entailment["verdicts"][0]["label"] == "ENTAILED"

    saved = await judge_saved_evaluation(
        _CombinedProvider(),
        "deepseek-v4-pro",
        store,
        [
            {
                "id": "q1",
                "query": "How is delivery confirmed?",
                "required_concepts": [["publisher confirms"]],
                "relevant_citations": [citation],
            }
        ],
        {
            "details": [
                {
                    "id": "q1",
                    "answer": f"RabbitMQ supports publisher confirms [{citation}].",
                }
            ]
        },
    )
    assert saved["judge"]["model"] == "deepseek-v4-pro"
    assert saved["judge"]["average_overall"] == 0.875
