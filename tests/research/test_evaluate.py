from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.research.evaluate import evaluate_retrieval, load_cases
from nanobot.research.store import ResearchStore


def test_evaluate_retrieval_reports_reproducible_metrics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "facts.md"
    source.write_text(
        "Publisher confirms tell a producer whether RabbitMQ accepted a message.\n\n"
        "Consumer acknowledgements tell RabbitMQ that processing completed.",
        encoding="utf-8",
    )
    store = ResearchStore(workspace)
    store.ingest_file(source, max_chars=220, overlap=0)
    publisher_citation = store.search("producer publisher confirms")[0]["citation"]
    consumer_citation = store.search("consumer processing completed")[0]["citation"]
    cases = [
        {
            "id": "publisher",
            "query": "How does a producer confirm acceptance?",
            "relevant_citations": [publisher_citation],
        },
        {
            "id": "consumer",
            "query": "How is completed consumer processing acknowledged?",
            "relevant_citations": [consumer_citation],
        },
    ]

    result = evaluate_retrieval(store, cases, top_k=2)

    assert result["cases"] == 2
    assert result["success_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] > 0


def test_load_cases_validates_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {"id": "q1", "query": "checkpoint", "relevant_citations": ["RF-DEADBEEF-1"]}
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_cases(dataset)[0]["id"] == "q1"

    dataset.write_text('{"query": "missing labels"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="relevant_citations"):
        load_cases(dataset)
