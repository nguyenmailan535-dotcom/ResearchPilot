"""ResearchFlow tools backed by the nanobot runtime."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.research.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DenseRetriever,
    FastEmbedReranker,
    RetrievalSuite,
)
from nanobot.research.store import ResearchStore


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


class _StoreHandle:
    """Lazily create one shared store so constructing AgentLoop stays side-effect free."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._store: ResearchStore | None = None
        self._retrieval: RetrievalSuite | None = None
        self._initialization_lock = threading.RLock()

    @property
    def store(self) -> ResearchStore:
        if self._store is None:
            with self._initialization_lock:
                if self._store is None:
                    self._store = ResearchStore(Path(self.workspace))
        return self._store

    @property
    def retrieval(self) -> RetrievalSuite:
        if self._retrieval is None:
            with self._initialization_lock:
                if self._retrieval is None:
                    model_name = os.getenv(
                        "RESEARCH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
                    )
                    dense = DenseRetriever(
                        self.store,
                        model_name=model_name,
                    )
                    backend = None
                    backend_name = os.getenv(
                        "RESEARCH_RETRIEVAL_BACKEND",
                        "milvus" if os.getenv("RESEARCH_MILVUS_URI") else "local",
                    ).strip().lower()
                    if backend_name == "milvus":
                        from nanobot.research.milvus import MilvusResearchBackend

                        backend = MilvusResearchBackend(
                            self.store,
                            model_name=model_name,
                            embedder=dense.embedder,
                        )
                    elif backend_name != "local":
                        raise ValueError(
                            "RESEARCH_RETRIEVAL_BACKEND must be either 'local' or 'milvus'"
                        )
                    reranker = None
                    if os.getenv("RESEARCH_ENABLE_RERANKER", "0").lower() in {
                        "1", "true", "yes",
                    }:
                        reranker = FastEmbedReranker(
                            os.getenv("RESEARCH_RERANKER_MODEL", "BAAI/bge-reranker-base")
                        )
                    self._retrieval = RetrievalSuite(
                        self.store, dense, backend=backend, reranker=reranker
                    )
        return self._retrieval


class _ResearchTool(Tool):
    def __init__(self, handle: _StoreHandle, allowed_dir: Path | None = None) -> None:
        self._handle = handle
        self.allowed_dir = Path(allowed_dir).expanduser().resolve() if allowed_dir else None

    @property
    def store(self) -> ResearchStore:
        return self._handle.store

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self._handle.workspace) / path
        resolved = path.resolve()
        if self.allowed_dir is not None:
            try:
                resolved.relative_to(self.allowed_dir)
            except ValueError as exc:
                raise ValueError(f"Path is outside the allowed workspace: {resolved}") from exc
        return resolved


class ResearchIngestTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_ingest"

    @property
    def description(self) -> str:
        return (
            "Index a local PDF, Markdown, text, source-code file, or a directory into the "
            "ResearchFlow evidence store. Re-indexing unchanged files is idempotent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path."},
                "max_files": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum files to scan for a directory.",
                },
                "chunk_size": {"type": "integer", "minimum": 200, "maximum": 4000},
                "chunk_overlap": {"type": "integer", "minimum": 0, "maximum": 1000},
                "chunk_strategy": {"type": "string", "enum": ["fixed", "structure"]},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        max_files: int = 100,
        chunk_size: int = 768,
        chunk_overlap: int = 120,
        chunk_strategy: str = "fixed",
    ) -> str:
        try:
            result = await asyncio.to_thread(
                self.store.ingest_path,
                self._resolve_path(path),
                max_files=max_files,
                max_chars=chunk_size,
                overlap=chunk_overlap,
                chunk_strategy=chunk_strategy,
            )
            return _json(result)
        except Exception as exc:
            return f"Error indexing research sources: {exc}"


class ResearchSearchTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_search"

    @property
    def description(self) -> str:
        return (
            "Search indexed research sources with BM25, dense vectors, RRF hybrid retrieval, "
            "optional query reflection and reranking. Returns stable citation IDs and may "
            "abstain when evidence is insufficient; cite claims as [RF-xxxxxxxx-N]."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "strategy": {
                    "type": "string",
                    "enum": ["auto", "bm25", "vector", "hybrid", "reflected_hybrid"],
                    "description": "auto is recommended and includes confidence-based reflection.",
                },
                "no_answer_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Abstain below this calibrated confidence threshold.",
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Rerank the fused candidate pool when a reranker is configured.",
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source IDs to restrict retrieval.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 6,
        source_ids: list[str] | None = None,
        strategy: str = "auto",
        no_answer_threshold: float | None = None,
        rerank: bool = False,
    ) -> str:
        try:
            outcome = await asyncio.to_thread(
                self._handle.retrieval.search,
                query,
                strategy=strategy,
                top_k=top_k,
                source_ids=source_ids,
                no_answer_threshold=no_answer_threshold,
                rerank=rerank,
            )
            if outcome.no_answer:
                return (
                    "INSUFFICIENT_EVIDENCE: the indexed corpus does not contain sufficiently "
                    f"strong evidence (strategy={outcome.used_strategy}, "
                    f"confidence={outcome.confidence:.4f}, latency_ms={outcome.latency_ms:.1f}). "
                    "Do not answer from model memory; refine the query or index more sources."
                )
            blocks = []
            for item in outcome.results:
                location = f", page {item['page']}" if item.get("page") else ""
                score_name = "reranker" if "reranker_score" in item else outcome.used_strategy
                blocks.append(
                    f"[{item['citation']}] {item['title']}{location} "
                    f"({score_name}={item['score']})\n{item['content']}"
                )
            metadata = (
                f"retrieval: requested={outcome.requested_strategy}; "
                f"used={outcome.used_strategy}; confidence={outcome.confidence:.4f}; "
                f"latency_ms={outcome.latency_ms:.1f}; degraded={str(outcome.degraded).lower()}; "
                f"reranked={str(outcome.reranked).lower()}"
            )
            if outcome.fallback_reason:
                metadata += f"; fallback={outcome.fallback_reason}"
            return "\n\n---\n\n".join(blocks) + f"\n\n---\n\n{metadata}"
        except Exception as exc:
            return f"Error searching research sources: {exc}"


class ResearchReadTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_read"

    @property
    def description(self) -> str:
        return "Read exact evidence chunks by their ResearchFlow citation IDs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Citation IDs such as RF-a1b2c3d4-1.",
                }
            },
            "required": ["citations"],
        }

    async def execute(self, citations: list[str]) -> str:
        try:
            chunks = self.store.get_chunks(citations)
            if not chunks:
                return "No valid evidence chunks found for those citation IDs."
            return _json(chunks)
        except Exception as exc:
            return f"Error reading research evidence: {exc}"


class ResearchMemoryTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_memory"

    @property
    def description(self) -> str:
        return (
            "Deprecated compatibility alias for research_decision. Store only evidence-backed "
            "project decisions and findings, not generic conversation memory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "search", "supersede"],
                },
                "project": {"type": "string", "description": "Stable project namespace."},
                "kind": {
                    "type": "string",
                    "enum": ["decision", "constraint", "finding", "reference", "preference", "profile"],
                },
                "content": {"type": "string"},
                "query": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "memory_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["action", "project"],
        }

    async def execute(
        self,
        action: str,
        project: str,
        kind: str = "reference",
        content: str = "",
        query: str = "",
        citations: list[str] | None = None,
        memory_id: str = "",
        limit: int = 10,
    ) -> str:
        try:
            if action == "remember":
                return _json(self.store.remember(project, kind, content, citations))
            if action == "search":
                return _json(self.store.search_memories(project, query, limit))
            if action == "supersede":
                if not memory_id:
                    return "Error: memory_id is required for supersede"
                return _json({"id": memory_id, "superseded": self.store.supersede_memory(memory_id)})
            return f"Error: unsupported memory action '{action}'"
        except Exception as exc:
            return f"Error managing research memory: {exc}"


class ResearchDecisionTool(ResearchMemoryTool):
    """Evidence ledger separated from nanobot's generic conversational memory."""

    @property
    def name(self) -> str:
        return "research_decision"

    @property
    def description(self) -> str:
        return (
            "Manage the project Evidence Ledger: durable decisions, constraints and findings "
            "linked to exact RF citations, with supersede history. Use nanobot memory for "
            "conversation context and user preferences."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["record", "search", "supersede"]},
                "project": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["decision", "constraint", "finding", "reference"],
                },
                "content": {"type": "string"},
                "query": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "entry_id": {"type": "string"},
                "supersedes_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["action", "project"],
        }

    async def execute(
        self,
        action: str,
        project: str,
        kind: str = "finding",
        content: str = "",
        query: str = "",
        citations: list[str] | None = None,
        entry_id: str = "",
        supersedes_id: str = "",
        limit: int = 10,
    ) -> str:
        try:
            if action == "record":
                return _json(
                    self.store.record_evidence_decision(
                        project,
                        kind,
                        content,
                        citations,
                        supersedes_id=supersedes_id or None,
                    )
                )
            if action == "search":
                return _json(self.store.search_evidence_decisions(project, query, limit))
            if action == "supersede":
                if not entry_id:
                    return "Error: entry_id is required for supersede"
                return _json(
                    {"id": entry_id, "superseded": self.store.supersede_evidence_decision(entry_id)}
                )
            return f"Error: unsupported Evidence Ledger action '{action}'"
        except Exception as exc:
            return f"Error managing Evidence Ledger: {exc}"


class ResearchReportTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_report"

    @property
    def description(self) -> str:
        return (
            "Verify ResearchFlow citations in a Markdown report or save a verified report under "
            "workspace/research/reports. Every evidence-backed claim should cite [RF-xxxxxxxx-N]."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["verify", "save"]},
                "content": {"type": "string", "description": "Markdown report content."},
                "path": {"type": "string", "description": "Existing Markdown report path."},
                "title": {"type": "string", "description": "Filename stem when saving."},
            },
            "required": ["action"],
        }

    async def execute(
        self, action: str, content: str = "", path: str = "", title: str = "research-report"
    ) -> str:
        try:
            if path:
                content = self._resolve_path(path).read_text(encoding="utf-8")
            if not content.strip():
                return "Error: content or path is required"
            if action == "verify":
                return _json(self.store.verify_report(content))
            if action == "save":
                return _json(self.store.save_report(title, content))
            return f"Error: unsupported report action '{action}'"
        except Exception as exc:
            return f"Error handling research report: {exc}"


class ResearchSourcesTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_sources"

    @property
    def description(self) -> str:
        return "List indexed sources and ResearchFlow store statistics."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self) -> str:
        try:
            return _json({"stats": self.store.stats(), "sources": self.store.list_sources()})
        except Exception as exc:
            return f"Error listing research sources: {exc}"


def register_research_tools(
    registry: ToolRegistry, workspace: Path, allowed_dir: Path | None = None
) -> None:
    """Register one coherent ResearchFlow toolset with a lazily shared store."""
    handle = _StoreHandle(workspace)
    for tool_cls in (
        ResearchIngestTool,
        ResearchSearchTool,
        ResearchReadTool,
        ResearchDecisionTool,
        ResearchMemoryTool,
        ResearchReportTool,
        ResearchSourcesTool,
    ):
        registry.register(tool_cls(handle, allowed_dir=allowed_dir))


def register_research_read_tools(
    registry: ToolRegistry, workspace: Path, allowed_dir: Path | None = None
) -> None:
    """Register the read-only ResearchFlow tools used by delegated researchers.

    Deliberately exclude ingest, report saving, memory and Evidence Ledger mutation. The
    coordinator remains the only writer while subagents search and inspect exact evidence.
    """
    handle = _StoreHandle(workspace)
    for tool_cls in (ResearchSourcesTool, ResearchSearchTool, ResearchReadTool):
        registry.register(tool_cls(handle, allowed_dir=allowed_dir))
