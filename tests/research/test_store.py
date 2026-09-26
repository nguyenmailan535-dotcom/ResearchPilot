from __future__ import annotations

from pathlib import Path

from nanobot.research.store import ResearchStore, split_text, tokenize


def test_multilingual_tokenizer_and_chunker() -> None:
    tokens = tokenize("Redis Streams 支持消费者组 consumer_group")
    assert "redis" in tokens
    assert "consumer_group" in tokens
    assert "消费" in tokens

    chunks = split_text("第一段内容。" * 80, max_chars=240, overlap=40)
    assert len(chunks) > 1
    assert all(1 <= len(chunk) <= 240 for chunk in chunks)

    malformed_pdf_text = split_text("valid text \ud800 after surrogate")
    assert malformed_pdf_text == ["valid text ? after surrogate"]


def test_ingest_is_idempotent_and_search_returns_stable_citations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "queues.md"
    source.write_text(
        "RabbitMQ supports publisher confirms and consumer acknowledgements.\n\n"
        "Redis Streams uses consumer groups and pending entry lists for delivery tracking.\n\n"
        "消息消费者应当通过业务唯一键保证重复投递时的幂等性。",
        encoding="utf-8",
    )
    store = ResearchStore(workspace)

    first = store.ingest_file(source, max_chars=220, overlap=30)
    second = store.ingest_file(source, max_chars=220, overlap=30)

    assert first["status"] == "indexed"
    assert second["status"] == "unchanged"
    assert second["chunks"] == first["chunks"]

    results = store.search("消费者 重复投递 幂等", top_k=3)
    assert results
    assert results[0]["citation"].startswith("RF-")
    assert "幂等" in results[0]["content"]

    citation = results[0]["citation"]
    exact = store.get_chunks([citation])
    assert exact[0]["citation"] == citation


def test_reindex_changed_file_replaces_old_chunks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "notes.txt"
    source.write_text("alpha evidence " * 30, encoding="utf-8")
    store = ResearchStore(workspace)
    first = store.ingest_file(source, max_chars=220)

    source.write_text("beta replacement " * 30, encoding="utf-8")
    updated = store.ingest_file(source, max_chars=220)

    assert updated["status"] == "updated"
    assert updated["source_id"] == first["source_id"]
    assert store.search("alpha") == []
    assert store.search("beta")


def test_report_verification_and_save(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.md"
    source.write_text("Checkpointing persists agent state after each completed tool call." * 4)
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("checkpointing agent state")[0]["citation"]

    report = (
        "# Result\n\n"
        f"Checkpointing preserves completed execution state and reduces repeated work "
        f"after a restart [{citation}]."
    )
    verification = store.verify_report(report)
    assert verification["valid"] is True
    assert verification["citation_coverage"] == 1.0

    invalid = store.verify_report("A sufficiently long unsupported paragraph " * 5 + "[RF-deadbeef-9]")
    assert invalid["valid"] is False
    assert invalid["missing"] == ["RF-DEADBEEF-9"]

    saved = store.save_report("checkpoint report", report)
    assert Path(saved["path"]).exists()
    assert saved["verification"]["valid"] is True


def test_project_memory_is_isolated_and_audited(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.md"
    source.write_text("RabbitMQ was selected because publisher confirms are required." * 3)
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("publisher confirms")[0]["citation"]

    memory = store.remember(
        "agent-runtime",
        "decision",
        "Use RabbitMQ for durable task delivery.",
        [citation],
    )

    assert store.search_memories("agent-runtime", "RabbitMQ")[0]["id"] == memory["id"]
    assert store.search_memories("another-project", "RabbitMQ") == []
    assert store.supersede_memory(memory["id"]) is True
    assert store.search_memories("agent-runtime", "RabbitMQ") == []
    assert store.stats()["events"] >= 4
