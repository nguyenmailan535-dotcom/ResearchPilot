"""ResearchFlow tools backed by the nanobot runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.research.store import ResearchStore


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


class _StoreHandle:
    """Lazily create one shared store so constructing AgentLoop stays side-effect free."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._store: ResearchStore | None = None

    @property
    def store(self) -> ResearchStore:
        if self._store is None:
            self._store = ResearchStore(Path(self.workspace))
        return self._store


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
            },
            "required": ["path"],
        }

    async def execute(self, path: str, max_files: int = 100) -> str:
        try:
            return _json(self.store.ingest_path(self._resolve_path(path), max_files=max_files))
        except Exception as exc:
            return f"Error indexing research sources: {exc}"


class ResearchSearchTool(_ResearchTool):
    @property
    def name(self) -> str:
        return "research_search"

    @property
    def description(self) -> str:
        return (
            "Search indexed research sources with multilingual BM25. Returns evidence chunks "
            "with stable citation IDs; cite claims as [RF-xxxxxxxx-N]."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source IDs to restrict retrieval.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, query: str, top_k: int = 6, source_ids: list[str] | None = None
    ) -> str:
        try:
            results = self.store.search(query, top_k=top_k, source_ids=source_ids)
            if not results:
                return "No matching evidence found. Broaden the query or index more sources."
            blocks = []
            for item in results:
                location = f", page {item['page']}" if item.get("page") else ""
                blocks.append(
                    f"[{item['citation']}] {item['title']}{location} "
                    f"(BM25={item['score']})\n{item['content']}"
                )
            return "\n\n---\n\n".join(blocks)
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
            "Manage durable project memory for stable decisions, constraints, preferences, and "
            "references. Do not store temporary reasoning or raw tool output."
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
                    "enum": ["decision", "constraint", "preference", "reference", "profile"],
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
        ResearchMemoryTool,
        ResearchReportTool,
        ResearchSourcesTool,
    ):
        registry.register(tool_cls(handle, allowed_dir=allowed_dir))
