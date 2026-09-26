from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.research.store import ResearchStore


def test_research_ingest_cli_reports_missing_path_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "research",
            "ingest",
            str(tmp_path / "missing-papers"),
            "--workspace",
            str(tmp_path / "workspace"),
        ],
    )

    assert result.exit_code == 1
    assert "ResearchFlow ingest failed" in result.output
    assert "Source path not found" in result.output
    assert "Traceback" not in result.output


def test_research_evaluate_cli_writes_machine_readable_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "facts.md"
    source.write_text(
        "Agent checkpoints preserve completed tool results across process restarts.",
        encoding="utf-8",
    )
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("checkpoints process restarts")[0]["citation"]

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "checkpoint",
                "query": "How are tool results preserved after a restart?",
                "relevant_citations": [citation],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"

    result = CliRunner().invoke(
        app,
        [
            "research",
            "evaluate",
            str(dataset),
            "--workspace",
            str(workspace),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["success_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
