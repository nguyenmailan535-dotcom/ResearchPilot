import numpy as np
import pytest

from nanobot.research.milvus import MilvusResearchBackend
from nanobot.research.retrieval import BGEM3DenseEmbedder
from nanobot.research.store import ResearchStore


class _BGEModel:
    def encode(self, texts, **kwargs):
        assert kwargs["return_dense"] is True
        assert kwargs["return_sparse"] is False
        return {"dense_vecs": np.ones((len(texts), 1024), dtype=np.float32)}


def test_bge_m3_adapter_uses_dense_representation_only() -> None:
    embedder = BGEM3DenseEmbedder(model=_BGEModel())
    vectors = list(embedder.passage_embed(["one", "two"]))
    assert len(vectors) == 2
    assert vectors[0].shape == (1024,)


def test_milvus_hit_mapping_and_source_filter(tmp_path) -> None:
    backend = MilvusResearchBackend(ResearchStore(tmp_path), embedder=object())
    result = backend._format_hits(
        [[{
            "id": "RF-12345678-1",
            "distance": 0.75,
            "entity": {
                "citation": "RF-12345678-1",
                "source_id": "abc123",
                "ordinal": 1,
                "page": 2,
                "title": "Paper",
                "path": "paper.pdf",
                "section": "Method",
                "element_type": "NarrativeText",
                "content": "evidence",
            },
        }]]
    )
    assert result[0]["citation"] == "RF-12345678-1"
    assert result[0]["score"] == 0.75
    assert backend._source_filter(["abc123"]) == 'source_id in ["abc123"]'
    assert backend._metadata_filter(
        ["abc123"],
        {
            "pages": [2, 3],
            "page_min": 2,
            "page_max": 5,
            "sections": ["Method"],
            "element_types": ["NarrativeText"],
        },
    ) == (
        'source_id in ["abc123"] and page in [2,3] and page >= 2 and page <= 5 '
        'and section in ["Method"] and element_type in ["NarrativeText"]'
    )
    assert backend._source_filter(['quoted"id']) == 'source_id in ["quoted\\\"id"]'
    with pytest.raises(ValueError):
        backend._source_filter(["bad\x00id"])
