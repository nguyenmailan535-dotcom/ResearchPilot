from __future__ import annotations

from pathlib import Path

import numpy as np

from nanobot.research.ablation import run_chunk_ablation
from nanobot.research.store import ResearchStore


class _Embedder:
    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()
        return np.asarray([float("rabbit" in lowered), float("redis" in lowered)], dtype=np.float32)

    def passage_embed(self, texts):
        yield from (self._vector(text) for text in texts)

    def query_embed(self, query):
        yield self._vector(query)


def test_chunk_ablation_reports_recommendation_and_latency(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.md"
    source.write_text("# Result\n\nRabbit acknowledgement provides durable delivery. " * 20)
    store = ResearchStore(workspace)
    store.ingest_file(source, max_chars=220, overlap=20)
    citation = store.search("rabbit acknowledgement")[0]["citation"]

    result = run_chunk_ablation(
        store,
        [{"id": "q1", "query": "rabbit acknowledgement", "relevant_citations": [citation]}],
        model_name="test/model",
        configs=[(256, 64, "fixed"), (384, 64, "structure")],
        embedder_factory=_Embedder,
    )

    assert result["recommended"]["recall_at_k"] == 1.0
    assert len(result["results"]) == 2
    assert result["results"][0]["p95_latency_ms"] >= 0
