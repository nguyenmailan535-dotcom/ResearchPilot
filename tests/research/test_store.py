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
    assert first["chunk_config"]["parser"] == {
        "name": "plain-text",
        "revision": "text-v1",
    }
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


def test_pdf_elements_are_packed_into_real_windows(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.pdf"
    source.write_bytes(b"synthetic pdf")
    store = ResearchStore(workspace)
    elements = [
        (1, f"paragraph {index} " + "evidence " * 8, "NarrativeText", "Method")
        for index in range(12)
    ]
    monkeypatch.setattr(ResearchStore, "_read_pdf", staticmethod(lambda _path: elements))

    result = store.ingest_file(source, max_chars=300, overlap=40)
    chunks = store.list_chunks()

    assert 3 < result["chunks"] < len(elements)
    assert all(len(chunk["content"]) <= 300 for chunk in chunks)
    assert all(chunk["page"] == 1 for chunk in chunks)
    assert any(chunk["element_type"] == "Composite" for chunk in chunks)
    assert result["chunk_config"]["parser"]["revision"] == "pdf-elements-packed-v2"


def test_semantic_pdf_chunking_preserves_section_boundaries(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.pdf"
    source.write_bytes(b"synthetic pdf")
    store = ResearchStore(workspace)
    elements = [
        (1, "method evidence " * 12, "NarrativeText", "Method"),
        (1, "method details " * 12, "NarrativeText", "Method"),
        (1, "result evidence " * 12, "NarrativeText", "Results"),
        (1, "result details " * 12, "NarrativeText", "Results"),
    ]
    monkeypatch.setattr(ResearchStore, "_read_pdf", staticmethod(lambda _path: elements))

    result = store.ingest_file(source, max_chars=300, overlap=40)
    chunks = store.list_chunks()

    assert result["chunk_config"]["strategy"] == "semantic"
    assert {chunk["section"] for chunk in chunks} == {"Method", "Results"}
    assert all(
        not ("method" in chunk["content"] and "result" in chunk["content"])
        for chunk in chunks
    )


def test_search_applies_page_section_and_element_metadata_filters(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.pdf"
    source.write_bytes(b"synthetic pdf")
    store = ResearchStore(workspace)
    elements = [
        (1, "shared evidence from the method section", "NarrativeText", "Method"),
        (2, "shared evidence from the results section", "Table", "Results"),
    ]
    monkeypatch.setattr(ResearchStore, "_read_pdf", staticmethod(lambda _path: elements))
    store.ingest_file(source, max_chars=300, overlap=0)

    filtered = store.search(
        "shared evidence",
        metadata_filter={
            "pages": [2],
            "sections": ["Results"],
            "element_types": ["Table"],
        },
    )

    assert len(filtered) == 1
    assert filtered[0]["page"] == 2
    assert filtered[0]["section"] == "Results"


def test_search_defaults_to_top_five(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ResearchStore(workspace)
    for index in range(6):
        source = workspace / f"paper-{index}.txt"
        source.write_text(f"shared retrieval evidence document {index}", encoding="utf-8")
        store.ingest_file(source)

    assert len(store.search("shared retrieval evidence")) == 5


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


def test_evidence_ledger_is_separate_versioned_and_citation_backed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.md"
    source.write_text("Hybrid retrieval combines lexical and semantic evidence." * 4)
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("hybrid retrieval lexical semantic")[0]["citation"]

    first = store.record_evidence_decision(
        "retrieval", "decision", "Use hybrid retrieval.", [citation]
    )
    second = store.record_evidence_decision(
        "retrieval",
        "decision",
        "Use hybrid retrieval followed by reranking.",
        [citation],
        supersedes_id=first["id"],
    )

    active = store.search_evidence_decisions("retrieval", "reranking")
    assert [item["id"] for item in active] == [second["id"]]
    assert store.stats()["active_evidence_decisions"] == 1
