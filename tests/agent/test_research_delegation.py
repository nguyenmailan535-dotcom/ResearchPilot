from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.research_delegate import ResearchDelegateTool
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.research.store import ResearchStore


def _manager(tmp_path: Path) -> SubagentManager:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())


def test_delegated_research_toolset_is_read_only(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    names = set(manager._build_research_tools().tool_names)

    assert {"research_sources", "research_search", "research_read"} == names
    assert not names.intersection(
        {"write_file", "edit_file", "exec", "research_ingest", "research_report", "research_decision"}
    )


@pytest.mark.asyncio
async def test_parallel_research_delegation_validates_citations(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("Hybrid retrieval combines lexical and dense evidence. " * 8, encoding="utf-8")
    store = ResearchStore(tmp_path)
    indexed = store.ingest_file(source, max_chars=300, overlap=40, chunk_strategy="fixed")
    citation = store.list_chunks(source_ids=[indexed["source_id"]])[0]["citation"]
    payload = {
        "summary": "Hybrid retrieval combines complementary signals.",
        "findings": [{"claim": "The corpus describes hybrid retrieval.", "citations": [citation]}],
        "gaps": [],
        "queries_used": ["hybrid retrieval"],
    }
    manager = _manager(tmp_path)
    manager.provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(payload),
            tool_calls=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
    )

    result = await manager.run_research_tasks(
        [
            {"label": "accuracy", "question": "What accuracy evidence is available?"},
            {"label": "memory", "question": "What memory evidence is available?"},
        ],
        max_concurrency=2,
    )

    assert result["mode"] == "parallel_read_only_research"
    assert result["task_count"] == 2
    assert all(item["status"] == "ok" for item in result["results"])
    assert all(item["valid_citations"] == [citation] for item in result["results"])
    assert all(item["invalid_citations"] == [] for item in result["results"])


@pytest.mark.asyncio
async def test_research_delegate_tool_returns_structured_json(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.run_research_tasks = AsyncMock(
        return_value={"mode": "parallel_read_only_research", "task_count": 2, "results": []}
    )
    tool = ResearchDelegateTool(manager)
    tasks = [
        {"label": "a", "question": "Find evidence for question A"},
        {"label": "b", "question": "Find evidence for question B"},
    ]

    output = await tool.execute(tasks, max_concurrency=2)

    assert json.loads(output)["task_count"] == 2
    manager.run_research_tasks.assert_awaited_once_with(tasks, max_concurrency=2)


@pytest.mark.asyncio
async def test_research_delegation_enforces_task_bounds(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    with pytest.raises(ValueError, match="2 or 3"):
        await manager.run_research_tasks(
            [{"label": "only", "question": "Only one question"}]
        )


@pytest.mark.asyncio
async def test_research_delegation_runs_independent_tasks_concurrently(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    async def delayed(label: str, question: str) -> dict:
        await asyncio.sleep(0.15)
        return {"label": label, "question": question, "status": "ok"}

    manager._run_research_task = AsyncMock(side_effect=delayed)
    tasks = [
        {"label": "a", "question": "Investigate independent question A"},
        {"label": "b", "question": "Investigate independent question B"},
        {"label": "c", "question": "Investigate independent question C"},
    ]

    started = time.perf_counter()
    result = await manager.run_research_tasks(tasks, max_concurrency=3)
    elapsed = time.perf_counter() - started

    assert result["task_count"] == 3
    assert elapsed < 0.30  # serial execution would take at least 0.45 seconds
