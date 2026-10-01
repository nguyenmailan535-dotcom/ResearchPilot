"""Milvus dense/BM25 retrieval backend for the production ResearchFlow profile."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

from nanobot.research.retrieval import DEFAULT_EMBEDDING_MODEL, BGEM3DenseEmbedder
from nanobot.research.store import ResearchStore


class MilvusResearchBackend:
    """Keep SQLite evidence rows synchronized with a Milvus hybrid-search collection.

    SQLite remains the source of truth for stable citations and workflow state.  Milvus is a
    rebuildable retrieval projection containing a BGE-M3 dense vector and a BM25-generated sparse
    vector for each chunk.
    """

    DENSE_DIMENSION = 1024

    def __init__(
        self,
        store: ResearchStore,
        *,
        uri: str | None = None,
        token: str | None = None,
        collection: str | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Any | None = None,
    ) -> None:
        self.store = store
        self.uri = uri or os.getenv("RESEARCH_MILVUS_URI", "http://127.0.0.1:19530")
        self.token = token if token is not None else os.getenv("RESEARCH_MILVUS_TOKEN", "")
        self.collection = collection or os.getenv(
            "RESEARCH_MILVUS_COLLECTION", "researchflow_chunks"
        )
        self.model_name = model_name
        self.embedder = embedder or BGEM3DenseEmbedder(model_name)
        self._client: Any | None = None
        self._sync_lock = threading.Lock()
        self._last_fingerprint: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise RuntimeError(
                    "Milvus retrieval requires the research dependencies: "
                    "pip install -e '.[research]'"
                ) from exc
            kwargs: dict[str, Any] = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
        return self._client

    @staticmethod
    def _content_hash(item: dict[str, Any]) -> str:
        payload = "\0".join(
            [
                str(item.get("citation", "")),
                str(item.get("content", "")),
                str(item.get("title", "")),
                str(item.get("page", "")),
                str(item.get("section", "")),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_collection(self) -> None:
        client = self.client
        if client.has_collection(self.collection):
            return
        from pymilvus import DataType, Function, FunctionType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="citation", datatype=DataType.VARCHAR, is_primary=True, max_length=64
        )
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="ordinal", datatype=DataType.INT64)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="element_type", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="content_hash", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.DENSE_DIMENSION,
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="researchflow_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["sparse_vector"],
            )
        )
        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name="dense_vector",
            index_name="dense_hnsw",
            index_type="HNSW",
            metric_type="IP",
            params={
                "M": int(os.getenv("RESEARCH_HNSW_M", "16")),
                "efConstruction": int(os.getenv("RESEARCH_HNSW_EF_CONSTRUCTION", "64")),
            },
        )
        indexes.add_index(
            field_name="sparse_vector",
            index_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )

    def _indexed_hashes(self) -> dict[str, str]:
        client = self.client
        iterator = client.query_iterator(
            collection_name=self.collection,
            batch_size=1000,
            filter="citation != ''",
            output_fields=["citation", "content_hash"],
        )
        result: dict[str, str] = {}
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                result.update(
                    (str(item["citation"]).upper(), str(item.get("content_hash", "")))
                    for item in batch
                )
        finally:
            iterator.close()
        return result

    def sync(self, *, force: bool = False) -> dict[str, int]:
        """Incrementally upsert changed chunks and remove stale Milvus points."""
        chunks = self.store.list_chunks()
        fingerprint = hashlib.sha256(
            "".join(
                f"{item['citation']}:{self._content_hash(item)};" for item in chunks
            ).encode("utf-8")
        ).hexdigest()
        if not force and fingerprint == self._last_fingerprint:
            return {"upserted": 0, "deleted": 0, "total": len(chunks)}
        with self._sync_lock:
            if not force and fingerprint == self._last_fingerprint:
                return {"upserted": 0, "deleted": 0, "total": len(chunks)}
            self._ensure_collection()
            indexed = self._indexed_hashes()
            local = {str(item["citation"]).upper(): self._content_hash(item) for item in chunks}
            changed = chunks if force else [
                item for item in chunks
                if indexed.get(str(item["citation"]).upper()) != local[str(item["citation"]).upper()]
            ]
            stale = sorted(set(indexed) - set(local))
            batch_size = max(1, int(os.getenv("RESEARCH_MILVUS_BATCH_SIZE", "32")))
            for offset in range(0, len(changed), batch_size):
                batch = changed[offset : offset + batch_size]
                vectors = np.asarray(
                    list(self.embedder.passage_embed(str(item["content"]) for item in batch)),
                    dtype=np.float32,
                )
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)
                rows = []
                for item, vector in zip(batch, vectors):
                    rows.append(
                        {
                            "citation": str(item["citation"]).upper(),
                            "source_id": str(item["source_id"]),
                            "ordinal": int(item["ordinal"]),
                            "page": int(item.get("page") or 0),
                            "title": str(item.get("title", ""))[:2048],
                            "path": str(item.get("path", ""))[:8192],
                            "section": str(item.get("section") or "")[:2048],
                            "element_type": str(item.get("element_type") or "")[:128],
                            "content_hash": local[str(item["citation"]).upper()],
                            "content": str(item["content"])[:65535],
                            "dense_vector": vector.tolist(),
                        }
                    )
                self.client.upsert(collection_name=self.collection, data=rows)
            if stale:
                for offset in range(0, len(stale), 500):
                    quoted = ",".join(f'"{value}"' for value in stale[offset : offset + 500])
                    self.client.delete(
                        collection_name=self.collection,
                        filter=f"citation in [{quoted}]",
                    )
            self.client.load_collection(self.collection)
            self._last_fingerprint = fingerprint
            return {"upserted": len(changed), "deleted": len(stale), "total": len(chunks)}

    @staticmethod
    def _source_filter(source_ids: Iterable[str] | None) -> str:
        return MilvusResearchBackend._metadata_filter(source_ids, None)

    @staticmethod
    def _metadata_filter(
        source_ids: Iterable[str] | None,
        metadata_filter: dict[str, Any] | None,
    ) -> str:
        """Translate supported evidence metadata into a safe Milvus expression."""
        metadata = metadata_filter or {}
        clauses: list[str] = []

        def quoted(values: Iterable[Any], label: str) -> str:
            normalized = [str(value) for value in values if str(value).strip()]
            if any("\x00" in value for value in normalized):
                raise ValueError(f"Invalid {label} filter")
            return ",".join(json.dumps(value, ensure_ascii=False) for value in normalized)

        selected = list(source_ids or metadata.get("source_ids") or [])
        if selected:
            clauses.append(f"source_id in [{quoted(selected, 'source_id')}]")
        pages = [int(value) for value in metadata.get("pages", [])]
        if pages:
            clauses.append("page in [{}]".format(",".join(str(value) for value in pages)))
        if metadata.get("page_min") is not None:
            clauses.append(f"page >= {int(metadata['page_min'])}")
        if metadata.get("page_max") is not None:
            clauses.append(f"page <= {int(metadata['page_max'])}")
        sections = metadata.get("sections", [])
        if sections:
            clauses.append(f"section in [{quoted(sections, 'section')}]")
        element_types = metadata.get("element_types", [])
        if element_types:
            clauses.append(
                f"element_type in [{quoted(element_types, 'element_type')}]"
            )
        return " and ".join(clauses)

    @staticmethod
    def _format_hits(raw: Any) -> list[dict[str, Any]]:
        hits = raw[0] if raw and isinstance(raw[0], list) else raw
        output: list[dict[str, Any]] = []
        for hit in hits or []:
            entity = dict(hit.get("entity") or {})
            page = int(entity.get("page") or 0)
            output.append(
                {
                    "citation": str(entity.get("citation") or hit.get("id", "")).upper(),
                    "source_id": str(entity.get("source_id", "")),
                    "ordinal": int(entity.get("ordinal") or 0),
                    "page": page or None,
                    "title": str(entity.get("title", "")),
                    "path": str(entity.get("path", "")),
                    "section": str(entity.get("section", "")),
                    "element_type": str(entity.get("element_type", "")),
                    "content": str(entity.get("content", "")),
                    "score": round(float(hit.get("distance", hit.get("score", 0.0))), 6),
                }
            )
        return output

    @property
    def _output_fields(self) -> list[str]:
        return [
            "citation", "source_id", "ordinal", "page", "title", "path", "section",
            "element_type", "content",
        ]

    def bm25(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.sync()
        raw = self.client.search(
            collection_name=self.collection,
            data=[query],
            anns_field="sparse_vector",
            limit=max(1, top_k),
            filter=self._metadata_filter(source_ids, metadata_filter),
            search_params={"metric_type": "BM25"},
            output_fields=self._output_fields,
        )
        return self._format_hits(raw)

    def vector(
        self,
        query: str,
        *,
        top_k: int,
        source_ids: Iterable[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.sync()
        vector = np.asarray(list(self.embedder.query_embed(query)), dtype=np.float32)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        raw = self.client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            anns_field="dense_vector",
            limit=max(1, top_k),
            filter=self._metadata_filter(source_ids, metadata_filter),
            search_params={
                "metric_type": "IP",
                "params": {"ef": int(os.getenv("RESEARCH_HNSW_EF", "64"))},
            },
            output_fields=self._output_fields,
        )
        return self._format_hits(raw)
