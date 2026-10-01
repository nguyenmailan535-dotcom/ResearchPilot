"""Dense, hybrid, and deterministic query-reflection retrieval for ResearchFlow."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanobot.research.store import ResearchStore, split_text, tokenize

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_STRATEGIES = ("auto", "bm25", "vector", "hybrid", "reflected_hybrid")
DEFAULT_NO_ANSWER_THRESHOLDS = {
    "bm25": 0.18,
    "vector": 0.55,
    "hybrid": 0.75,
    "reflected_hybrid": 0.75,
}
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "these", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}

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


class BGEM3DenseEmbedder:
    """FlagEmbedding adapter exposing the small interface used by ResearchFlow.

    BGE-M3 can also emit lexical and ColBERT representations.  ResearchFlow deliberately uses
    only its normalized dense representation here: Milvus' BM25 Function owns the sparse path,
    so the two retrievers remain independently measurable in the existing ablation suite.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model
        self._model_lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from FlagEmbedding import BGEM3FlagModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "BGE-M3 requires the research dependencies: "
                            "pip install -e '.[research]'"
                        ) from exc
                    device = os.getenv("RESEARCH_BGE_DEVICE", "cpu")
                    use_fp16 = os.getenv("RESEARCH_BGE_USE_FP16", "0").lower() in {
                        "1", "true", "yes",
                    }
                    self._model = BGEM3FlagModel(
                        self.model_name,
                        use_fp16=use_fp16,
                        devices=device,
                    )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        output = self.model.encode(
            texts,
            batch_size=max(1, int(os.getenv("RESEARCH_EMBEDDING_BATCH_SIZE", "8"))),
            max_length=max(128, int(os.getenv("RESEARCH_EMBEDDING_MAX_LENGTH", "1024"))),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(output["dense_vecs"], dtype=np.float32)

    def passage_embed(self, passages: Iterable[str]) -> Iterable[np.ndarray]:
        return iter(self._encode(list(passages)))

    def query_embed(self, query: str | Iterable[str]) -> Iterable[np.ndarray]:
        values = [query] if isinstance(query, str) else list(query)
        return iter(self._encode(values))


class DenseRetriever:
    """FastEmbed-backed cosine retriever with a corpus-addressed on-disk cache."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Any | None = None,
        segment_chars: int = 768,
        segment_overlap: int = 120,
    ) -> None:
        self.store = store
        self.model_name = model_name
        self._embedder = embedder
        self.segment_chars = segment_chars
        self.segment_overlap = segment_overlap
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._segment_chunk_indices: np.ndarray | None = None
        self._build_lock = threading.Lock()

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            if self.model_name.lower() in {"baai/bge-m3", "bge-m3"}:
                self._embedder = BGEM3DenseEmbedder(self.model_name)
                return self._embedder
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

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._embeddings is None:
            # Delegated researchers share one retriever and may reach the first dense query at
            # the same time. Build/load the cache once, then keep query execution lock-free.
            with self._build_lock:
                if self._embeddings is None:
                    self.build_index()
        assert self._embeddings is not None
        assert self._segment_chunk_indices is not None
        query_vector = np.asarray(list(self.embedder.query_embed(query)), dtype=np.float32)
        query_vector = self._normalize(query_vector)[0]
        segment_scores = self._embeddings @ query_vector
        scores = np.full(len(self._chunks), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self._segment_chunk_indices, segment_scores)
        metadata = metadata_filter or {}
        selected = {
            str(value)
            for value in (source_ids or metadata.get("source_ids") or [])
            if value
        }
        pages = {int(value) for value in metadata.get("pages", [])}
        sections = {str(value) for value in metadata.get("sections", []) if value}
        element_types = {
            str(value) for value in metadata.get("element_types", []) if value
        }
        page_min = metadata.get("page_min")
        page_max = metadata.get("page_max")

        def matches(chunk: dict[str, Any]) -> bool:
            page = chunk.get("page")
            return (
                (not selected or chunk["source_id"] in selected)
                and (not pages or page in pages)
                and (page_min is None or (page is not None and page >= int(page_min)))
                and (page_max is None or (page is not None and page <= int(page_max)))
                and (not sections or chunk.get("section") in sections)
                and (not element_types or chunk.get("element_type") in element_types)
            )

        candidates = np.asarray(
            [
                index
                for index, chunk in enumerate(self._chunks)
                if matches(chunk)
            ],
            dtype=np.int32,
        )
        if not len(candidates):
            return []
        count = max(1, min(top_k, len(candidates)))
        best = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
        output: list[dict[str, Any]] = []
        for index in best:
            item = dict(self._chunks[int(index)])
            item["score"] = round(float(scores[int(index)]), 6)
            output.append(item)
        return output


class FastEmbedReranker:
    """Optional cross-encoder adapter; loaded only when reranking is requested."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Reranking requires the optional dependency: pip install -e '.[research]'"
                ) from exc
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(value) for value in self.model.rerank(query, passages)]


@dataclass(slots=True)
class RetrievalOutcome:
    """Observable result for online retrieval, fallback and abstention decisions."""

    query: str
    requested_strategy: str
    used_strategy: str
    results: list[dict[str, Any]]
    confidence: float
    no_answer: bool
    degraded: bool = False
    fallback_reason: str | None = None
    reflected_query: str | None = None
    reranked: bool = False
    latency_ms: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reflect_query(query: str, seed_results: list[dict[str, Any]]) -> str:
    """Deterministically rewrite a weak query using glossary and first-pass metadata."""
    lowered = query.lower()
    additions: list[str] = []
    for triggers, expansion in _REFLECTION_GLOSSARY:
        if any(trigger.lower() in lowered for trigger in triggers):
            additions.append(expansion)
    # For queries outside the curated glossary, use only compact title/section metadata from
    # the strongest first-pass evidence. This produces an observable second query without
    # copying passage claims or requiring an online LLM in the retrieval path.
    if not additions and seed_results:
        query_terms = set(tokenize(query))
        feedback_terms: list[str] = []
        for item in seed_results[:3]:
            metadata = f"{item.get('title', '')} {item.get('section', '')}"
            for term in tokenize(metadata):
                if (
                    len(term) > 2
                    and term not in query_terms
                    and term not in _QUERY_STOPWORDS
                    and term not in feedback_terms
                ):
                    feedback_terms.append(term)
                if len(feedback_terms) >= 6:
                    break
            if len(feedback_terms) >= 6:
                break
        if feedback_terms:
            additions.append(" ".join(feedback_terms))
    return " ".join([query, *dict.fromkeys(additions)]).strip()


class RetrievalSuite:
    """Comparable BM25, dense, hybrid, and reflected-hybrid retrieval methods."""

    def __init__(
        self,
        store: ResearchStore,
        dense: DenseRetriever,
        *,
        backend: Any | None = None,
        reranker: Any | None = None,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        rank_constant: int | None = None,
    ) -> None:
        self.store = store
        self.dense = dense
        self.backend = backend
        self.reranker = reranker
        self.bm25_weight = (
            float(os.getenv("RESEARCH_RRF_BM25_WEIGHT", "1.0"))
            if bm25_weight is None
            else bm25_weight
        )
        self.vector_weight = (
            float(os.getenv("RESEARCH_RRF_VECTOR_WEIGHT", "1.0"))
            if vector_weight is None
            else vector_weight
        )
        self.rank_constant = (
            int(os.getenv("RESEARCH_RRF_RANK_CONSTANT", "20"))
            if rank_constant is None
            else rank_constant
        )

    def bm25(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.backend is not None:
            return self.backend.bm25(
                query,
                top_k=top_k,
                source_ids=source_ids,
                metadata_filter=metadata_filter,
            )
        return self.store.search(
            query,
            top_k=top_k,
            source_ids=source_ids,
            metadata_filter=metadata_filter,
        )

    def vector(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.backend is not None:
            return self.backend.vector(
                query,
                top_k=top_k,
                source_ids=source_ids,
                metadata_filter=metadata_filter,
            )
        return self.dense.search(
            query,
            top_k=top_k,
            source_ids=source_ids,
            metadata_filter=metadata_filter,
        )

    def hybrid(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pool = max(20, top_k * 4)
        bm25 = self.bm25(
            query, top_k=pool, source_ids=source_ids, metadata_filter=metadata_filter
        )
        vector = self.vector(
            query, top_k=pool, source_ids=source_ids, metadata_filter=metadata_filter
        )
        fused = reciprocal_rank_fusion(
            [bm25, vector],
            top_k=top_k,
            rank_constant=self.rank_constant,
            weights=(self.bm25_weight, self.vector_weight),
        )
        bm25_ids = {item["citation"] for item in bm25}
        vector_ids = {item["citation"] for item in vector}
        for item in fused:
            item["retrievers"] = [
                name
                for name, ids in (("bm25", bm25_ids), ("vector", vector_ids))
                if item["citation"] in ids
            ]
        return fused

    def reflected_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pool = max(20, top_k * 4)
        seed = self.hybrid(
            query, top_k=pool, source_ids=source_ids, metadata_filter=metadata_filter
        )
        reflected = reflect_query(query, seed)
        second_pass = self.hybrid(
            reflected,
            top_k=pool,
            source_ids=source_ids,
            metadata_filter=metadata_filter,
        )
        return reciprocal_rank_fusion([seed, second_pass], top_k=top_k)

    @staticmethod
    def _confidence(
        strategy: str,
        results: list[dict[str, Any]],
        query: str = "",
    ) -> tuple[float, dict[str, float]]:
        if not results:
            return 0.0, {"top_score": 0.0, "margin": 0.0, "agreement": 0.0}
        top = float(results[0].get("score", 0.0))
        second = float(results[1].get("score", 0.0)) if len(results) > 1 else 0.0
        query_tokens = {
            token for token in tokenize(query) if len(token) > 1 and token not in _QUERY_STOPWORDS
        }
        evidence_tokens = set(
            token
            for item in results[:5]
            for token in tokenize(str(item.get("content", "")))
            if len(token) > 1
        )
        query_coverage = len(query_tokens & evidence_tokens) / max(1, len(query_tokens))
        if strategy == "bm25":
            strength = 1.0 - math.exp(-max(0.0, top))
            margin = max(0.0, top - second) / max(top, 1e-9)
            confidence = (0.8 * strength + 0.2 * margin) * (0.6 + 0.4 * query_coverage)
        elif strategy == "vector":
            strength = max(0.0, min(1.0, (top + 1.0) / 2.0))
            margin = max(0.0, top - second)
            confidence = (0.85 * strength + 0.15 * min(1.0, margin * 4)) * (
                0.6 + 0.4 * query_coverage
            )
        else:
            agreement = sum(
                1 for item in results[:5] if len(item.get("retrievers", [])) >= 2
            ) / min(5, len(results))
            # RRF scores are deliberately small; agreement is the stable cross-method signal.
            strength = min(1.0, top * 25.0)
            margin = max(0.0, top - second) * 100.0
            confidence = (
                0.45 * strength + 0.45 * agreement + 0.10 * min(1.0, margin)
            ) * (0.6 + 0.4 * query_coverage)
            return round(confidence, 4), {
                "top_score": round(top, 6),
                "margin": round(margin, 6),
                "agreement": round(agreement, 4),
                "query_coverage": round(query_coverage, 4),
            }
        return round(confidence, 4), {
            "top_score": round(top, 6),
            "margin": round(margin, 6),
            "agreement": 0.0,
            "query_coverage": round(query_coverage, 4),
        }

    def _rerank(
        self, query: str, results: list[dict[str, Any]], *, top_k: int
    ) -> list[dict[str, Any]]:
        if not results or self.reranker is None:
            return results[:top_k]
        passages = [str(item["content"]) for item in results]
        if hasattr(self.reranker, "scores"):
            scores = self.reranker.scores(query, passages)
        else:
            scores = self.reranker(query, passages)
        if len(scores) != len(results):
            raise ValueError("Reranker returned a different number of scores than candidates")
        ranked: list[dict[str, Any]] = []
        for item, score in zip(results, scores):
            enriched = dict(item)
            enriched["retrieval_score"] = enriched.get("score")
            enriched["reranker_score"] = round(float(score), 6)
            enriched["score"] = round(float(score), 6)
            ranked.append(enriched)
        ranked.sort(key=lambda item: (-item["reranker_score"], item["citation"]))
        return ranked[:top_k]

    def search(
        self,
        query: str,
        *,
        strategy: str = "auto",
        top_k: int = 5,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        no_answer_threshold: float | None = None,
        allow_fallback: bool = True,
        rerank: bool = False,
    ) -> RetrievalOutcome:
        """Run online retrieval with observable fallback and calibrated abstention metadata."""
        if strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        started = time.perf_counter()
        requested = strategy
        used = "hybrid" if strategy == "auto" else strategy
        env_threshold = os.getenv("RESEARCH_NO_ANSWER_THRESHOLD")
        threshold = (
            float(env_threshold)
            if no_answer_threshold is None and env_threshold is not None
            else DEFAULT_NO_ANSWER_THRESHOLDS[used]
            if no_answer_threshold is None
            else max(0.0, min(1.0, float(no_answer_threshold)))
        )
        degraded = False
        fallback_reason: str | None = None
        reflected_query: str | None = None
        pool = max(20, top_k * 4) if rerank else top_k

        try:
            results = getattr(self, used)(
                query,
                top_k=pool,
                source_ids=source_ids,
                metadata_filter=metadata_filter,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            if not allow_fallback:
                raise
            if used == "bm25":
                if self.backend is None:
                    raise
                # If the external retrieval plane itself is unavailable, SQLite FTS remains a
                # useful local sparse fallback.  This keeps the Agent operational while exposing
                # the degradation in RetrievalOutcome rather than silently changing semantics.
                results = self.store.search(
                    query,
                    top_k=pool,
                    source_ids=source_ids,
                    metadata_filter=metadata_filter,
                )
                fallback_reason = f"sparse backend unavailable: {exc}"
            else:
                try:
                    results = self.bm25(
                        query,
                        top_k=pool,
                        source_ids=source_ids,
                        metadata_filter=metadata_filter,
                    )
                    fallback_reason = f"dense retrieval unavailable: {exc}"
                except (RuntimeError, OSError, ValueError) as backend_exc:
                    results = self.store.search(
                        query,
                        top_k=pool,
                        source_ids=source_ids,
                        metadata_filter=metadata_filter,
                    )
                    fallback_reason = (
                        f"dense retrieval unavailable: {exc}; "
                        f"sparse backend unavailable: {backend_exc}"
                    )
            used = "bm25"
            degraded = True

        confidence, signals = self._confidence(used, results, query)
        if requested == "auto" and confidence < threshold and used != "bm25":
            try:
                reflected_query = reflect_query(query, results)
                if reflected_query != query:
                    reflected = self.reflected_hybrid(
                        query,
                        top_k=pool,
                        source_ids=source_ids,
                        metadata_filter=metadata_filter,
                    )
                    reflected_confidence, reflected_signals = self._confidence(
                        "reflected_hybrid", reflected, query
                    )
                    if reflected_confidence >= confidence:
                        results = reflected
                        confidence = reflected_confidence
                        signals = reflected_signals
                        used = "reflected_hybrid"
            except (RuntimeError, OSError, ValueError) as exc:
                degraded = True
                fallback_reason = fallback_reason or f"query reflection unavailable: {exc}"

        reranked = bool(rerank and self.reranker is not None and results)
        if reranked:
            results = self._rerank(query, results, top_k=top_k)
            confidence, rerank_signals = self._confidence("vector", results, query)
            signals.update({f"reranker_{key}": value for key, value in rerank_signals.items()})
        else:
            results = results[:top_k]
        no_answer = not results or confidence < threshold
        return RetrievalOutcome(
            query=query,
            requested_strategy=requested,
            used_strategy=used,
            results=results,
            confidence=confidence,
            no_answer=no_answer,
            degraded=degraded,
            fallback_reason=fallback_reason,
            reflected_query=reflected_query,
            reranked=reranked,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            signals=signals,
        )
