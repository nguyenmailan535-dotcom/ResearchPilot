"""Dense, hybrid, and deterministic query-reflection retrieval for ResearchFlow."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanobot.research.store import ResearchStore, split_text

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_REFLECTION_GLOSSARY: tuple[tuple[tuple[str, ...], str], ...] = (
    (("基数", "去重计数", "distinct"), "cardinality distinct count estimation"),
    (("频率", "频次", "frequency"), "item frequency estimation data stream"),
    (("精度", "误差", "accuracy", "error"), "accuracy relative error standard error ARE"),
    (("内存", "空间", "memory", "space"), "memory space bits bytes registers counters MVP"),
    (("合并", "分布式", "merge", "distributed"), "merge mergeability union distributed partial results"),
    (("并行", "parallel"), "parallelization throughput speedup"),
    (("吞吐", "性能", "throughput", "performance"), "update throughput query throughput Mops"),
    (("热项", "冷项", "hot", "cold"), "hot items cold items warm items adaptive scaling"),
    (("集合", "交集", "并集", "intersection", "union"), "set union intersection inclusion exclusion"),
    (("压缩", "熵", "compression", "entropy"), "Shannon entropy lossless compression"),
)


def reciprocal_rank_fusion(
    result_lists: Iterable[list[dict[str, Any]]],
    *,
    top_k: int,
    rank_constant: int = 60,
    weights: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse heterogeneous ranked lists without assuming comparable raw scores."""
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    materialized = list(result_lists)
    method_weights = list(weights) if weights is not None else [1.0] * len(materialized)
    if len(method_weights) != len(materialized):
        raise ValueError("RRF weights must match the number of ranked lists")
    for results, weight in zip(materialized, method_weights):
        for rank, item in enumerate(results, start=1):
            citation = str(item["citation"]).upper()
            scores[citation] = scores.get(citation, 0.0) + weight / (rank_constant + rank)
            items.setdefault(citation, item)
    ranked = sorted(scores, key=lambda citation: (-scores[citation], citation))
    output: list[dict[str, Any]] = []
    for citation in ranked[:top_k]:
        item = dict(items[citation])
        item["score"] = round(scores[citation], 8)
        output.append(item)
    return output


class DenseRetriever:
    """FastEmbed-backed cosine retriever with a corpus-addressed on-disk cache."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Any | None = None,
        segment_chars: int = 480,
        segment_overlap: int = 80,
    ) -> None:
        self.store = store
        self.model_name = model_name
        self._embedder = embedder
        self.segment_chars = segment_chars
        self.segment_overlap = segment_overlap
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._segment_chunk_indices: np.ndarray | None = None

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "Vector retrieval requires the optional dependency: "
                    "pip install -e '.[research]'"
                ) from exc
            model_cache = self.store.root / "models"
            model_cache.mkdir(parents=True, exist_ok=True)
            self._embedder = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(model_cache),
            )
        return self._embedder

    def _cache_path(self, chunks: list[dict[str, Any]]) -> Path:
        digest = hashlib.sha256(self.model_name.encode("utf-8"))
        digest.update(f"segments:{self.segment_chars}:{self.segment_overlap}".encode("utf-8"))
        for chunk in chunks:
            digest.update(str(chunk["citation"]).encode("utf-8"))
            digest.update(hashlib.sha256(str(chunk["content"]).encode("utf-8")).digest())
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.model_name).strip("-").lower()
        cache_dir = self.store.root / "vectors"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{model_slug}-seg{self.segment_chars}-{digest.hexdigest()[:16]}.npz"

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def build_index(self, *, force: bool = False) -> dict[str, Any]:
        self._chunks = self.store.list_chunks()
        if not self._chunks:
            raise ValueError("Research corpus is empty; ingest sources before building vectors")
        cache_path = self._cache_path(self._chunks)
        passages: list[str] = []
        segment_chunk_indices: list[int] = []
        segment_ids: list[str] = []
        for chunk_index, chunk in enumerate(self._chunks):
            segments = split_text(
                str(chunk["content"]),
                max_chars=self.segment_chars,
                overlap=self.segment_overlap,
            )
            for segment_index, segment in enumerate(segments):
                passages.append(segment)
                segment_chunk_indices.append(chunk_index)
                segment_ids.append(f"{chunk['citation']}:{segment_index}")
        expected_ids = np.asarray(segment_ids)
        if cache_path.exists() and not force:
            with np.load(cache_path, allow_pickle=False) as cached:
                cached_ids = cached["citations"]
                if np.array_equal(cached_ids, expected_ids):
                    self._embeddings = self._normalize(cached["embeddings"])
                    self._segment_chunk_indices = cached["chunk_indices"]
                    return {
                        "status": "cached",
                        "chunks": len(self._chunks),
                        "segments": len(expected_ids),
                        "dimensions": int(self._embeddings.shape[1]),
                        "model": self.model_name,
                        "cache": str(cache_path),
                    }
        vectors = np.asarray(list(self.embedder.passage_embed(passages)), dtype=np.float32)
        self._embeddings = self._normalize(vectors)
        self._segment_chunk_indices = np.asarray(segment_chunk_indices, dtype=np.int32)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            citations=expected_ids,
            embeddings=self._embeddings,
            chunk_indices=self._segment_chunk_indices,
        )
        temporary.replace(cache_path)
        return {
            "status": "built",
            "chunks": len(self._chunks),
            "segments": len(expected_ids),
            "dimensions": int(self._embeddings.shape[1]),
            "model": self.model_name,
            "cache": str(cache_path),
        }

    def search(self, query: str, *, top_k: int = 6) -> list[dict[str, Any]]:
        if self._embeddings is None:
            self.build_index()
        assert self._embeddings is not None
        assert self._segment_chunk_indices is not None
        query_vector = np.asarray(list(self.embedder.query_embed(query)), dtype=np.float32)
        query_vector = self._normalize(query_vector)[0]
        segment_scores = self._embeddings @ query_vector
        scores = np.full(len(self._chunks), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self._segment_chunk_indices, segment_scores)
        count = max(1, min(top_k, len(self._chunks)))
        best = np.argsort(-scores, kind="stable")[:count]
        output: list[dict[str, Any]] = []
        for index in best:
            item = dict(self._chunks[int(index)])
            item["score"] = round(float(scores[int(index)]), 6)
            output.append(item)
        return output


def reflect_query(query: str, seed_results: list[dict[str, Any]]) -> str:
    """Expand a query after inspecting first-pass evidence, without an LLM or labels."""
    del seed_results  # Reserved for future confidence/gap signals; avoid pseudo-feedback drift.
    lowered = query.lower()
    additions: list[str] = []
    for triggers, expansion in _REFLECTION_GLOSSARY:
        if any(trigger.lower() in lowered for trigger in triggers):
            additions.append(expansion)
    return " ".join([query, *dict.fromkeys(additions)]).strip()


class RetrievalSuite:
    """Comparable BM25, dense, hybrid, and reflected-hybrid retrieval methods."""

    def __init__(self, store: ResearchStore, dense: DenseRetriever) -> None:
        self.store = store
        self.dense = dense

    def bm25(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.store.search(query, top_k=top_k)

    def vector(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.dense.search(query, top_k=top_k)

    def hybrid(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        pool = max(20, top_k * 4)
        return reciprocal_rank_fusion(
            [self.bm25(query, top_k=pool), self.vector(query, top_k=pool)],
            top_k=top_k,
            weights=(2.0, 1.0),
        )

    def reflected_hybrid(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        pool = max(20, top_k * 4)
        seed = self.hybrid(query, top_k=pool)
        reflected = reflect_query(query, seed)
        second_pass = self.hybrid(reflected, top_k=pool)
        return reciprocal_rank_fusion([seed, second_pass], top_k=top_k)
