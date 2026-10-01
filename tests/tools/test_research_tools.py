from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.research import register_research_read_tools, register_research_tools


def _registry(workspace: Path, *, restricted: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    register_research_tools(
        registry,
        workspace,
        allowed_dir=workspace if restricted else None,
    )
    return registry


@pytest.mark.asyncio
async def test_research_tools_complete_grounded_workflow(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "agent.md"
    source.write_text(
        "A durable agent stores a checkpoint after a successful tool call. " * 5,
        encoding="utf-8",
    )
    registry = _registry(workspace)

    ingest = await registry.execute("research_ingest", {"path": str(source)})
    assert '"status": "indexed"' in ingest

    search = await registry.execute(
        "research_search",
        {"query": "durable checkpoint tool call", "top_k": 3, "strategy": "bm25"},
    )
    assert "RF-" in search
    citation = search.split("[", 1)[1].split("]", 1)[0]

    memory = await registry.execute(
        "research_memory",
        {
            "action": "remember",
            "project": "runtime",
            "kind": "decision",
            "content": "Persist a checkpoint after successful tools.",
            "citations": [citation],
        },
    )
    assert "RM-" in memory

    decision = await registry.execute(
        "research_decision",
        {
            "action": "record",
            "project": "runtime",
            "kind": "decision",
            "content": "Use evidence-backed checkpoints.",
            "citations": [citation],
        },
    )
    assert "ED-" in decision

    report = await registry.execute(
        "research_report",
        {
            "action": "verify",
            "content": "A durable runtime checkpoints successful tool calls "
            + "to avoid repeated work after restart "
            + f"[{citation}].",
        },
    )
    assert '"valid": true' in report


@pytest.mark.asyncio
async def test_research_ingest_respects_workspace_restriction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    registry = _registry(workspace)

    result = await registry.execute("research_ingest", {"path": str(outside)})

    assert result.startswith("Error indexing research sources")
    assert "outside the allowed workspace" in result


@pytest.mark.asyncio
async def test_research_search_can_route_insufficient_local_evidence_to_web(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.txt"
    source.write_text("local evidence about cardinality estimation", encoding="utf-8")
    web_search = SimpleNamespace(
        execute=AsyncMock(return_value="Results for: missing topic\n\n1. External paper")
    )
    registry = ToolRegistry()
    register_research_tools(
        registry,
        workspace,
        allowed_dir=workspace,
        web_search_tool=web_search,
    )
    await registry.execute("research_ingest", {"path": str(source)})

    result = await registry.execute(
        "research_search",
        {
            "query": "missing topic",
            "strategy": "bm25",
            "no_answer_threshold": 0.9,
            "allow_web_fallback": True,
        },
    )

    assert result.startswith("EXTERNAL_WEB_FALLBACK")
    assert "not RF citations" in result
    web_search.execute.assert_awaited_once()


def test_registers_expected_research_tools_without_creating_database(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry = _registry(workspace, restricted=False)

    assert {
        "research_ingest",
        "research_search",
        "research_decision",
        "research_read",
        "research_memory",
        "research_report",
        "research_sources",
    }.issubset(registry.tool_names)
    assert not (workspace / "research" / "research.db").exists()


def test_read_only_research_tools_exclude_all_mutations(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_research_read_tools(registry, tmp_path)

    assert set(registry.tool_names) == {
        "research_sources",
        "research_search",
        "research_read",
    }
    for forbidden in (
        "research_ingest",
        "research_report",
        "research_decision",
        "research_memory",
    ):
        assert not registry.has(forbidden)
