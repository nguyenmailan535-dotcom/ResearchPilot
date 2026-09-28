from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.research.end_to_end import (
    aggregate_evaluation_details,
    evaluate_agent_answers,
    load_end_to_end_cases,
    score_answer,
)
from nanobot.research.store import ResearchStore


def _case(citation: str) -> dict:
    return {
        "id": "q1",
        "query": "What is the error?",
        "required_concepts": [["1.04"], ["square root", "√m"]],
        "relevant_citations": [citation],
        "language": "en",
    }


def _store_with_source(tmp_path: Path) -> tuple[ResearchStore, str]:
    source = tmp_path / "paper.txt"
    source.write_text("The standard error is 1.04 divided by the square root of m.", encoding="utf-8")
    store = ResearchStore(tmp_path / "workspace")
    indexed = store.ingest_file(source)
    return store, f"RF-{indexed['source_id'][:8]}-1"


def test_load_end_to_end_cases(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"query":"q","required_concepts":[["a","b"],"c"],'
        '"relevant_citations":["RF-12345678-1"]}\n',
        encoding="utf-8",
    )
    cases = load_end_to_end_cases(dataset)
    assert cases[0]["required_concepts"] == [["a", "b"], ["c"]]
    assert cases[0]["id"] == "1"


def test_score_answer_checks_concepts_and_citations(tmp_path: Path) -> None:
    store, citation = _store_with_source(tmp_path)
    score = score_answer(
        store,
        _case(citation),
        f"The error is 1.04 over the square root of m [{citation}].",
    )
    assert score["completed"] is True
    assert score["answer_pass"] is True
    assert score["concept_coverage"] == 1.0
    assert score["citation_validity"] == 1.0
    assert score["relevant_citation_recall"] == 1.0


@pytest.mark.asyncio
async def test_evaluate_agent_answers_aggregates_runtime_metrics(tmp_path: Path) -> None:
    store, citation = _store_with_source(tmp_path)

    class FakeAgent:
        _last_usage = {"prompt_tokens": 100, "completion_tokens": 20}

        async def process_direct(self, **kwargs):
            await kwargs["on_progress"]("research_search", tool_hint=True)
            await kwargs["on_stream"]("answer")
            return SimpleNamespace(
                content=f"The result is 1.04 divided by √m [{citation}]."
            )

    result = await evaluate_agent_answers(FakeAgent(), store, [_case(citation)])
    assert result["cases"] == 1
    assert result["metrics"]["task_completion_rate"] == 1.0
    assert result["metrics"]["answer_pass_rate"] == 1.0
    assert result["metrics"]["average_prompt_tokens"] == 100.0
    assert result["metrics"]["average_tool_events"] == 1.0
    resumed = aggregate_evaluation_details(result["details"])
    assert resumed["metrics"] == result["metrics"]
