"""Persistent evidence store used by the ResearchFlow tools.

The implementation intentionally stays framework-free.  It uses SQLite for
durability and a small multilingual BM25 implementation for deterministic,
offline retrieval.  This keeps the retrieval path inspectable in interviews
and avoids coupling nanobot's runtime to a vector database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from nanobot.utils.helpers import ensure_dir, safe_filename

SUPPORTED_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".java",
    ".json",
    ".md",
    ".pdf",
    ".py",
    ".rst",
    ".text",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

_LATIN_OR_CJK = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+")
_CITATION = re.compile(r"\[(RF-[0-9a-f]{8}-\d+)\]", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenize(text: str) -> list[str]:
    """Tokenize English identifiers and Chinese text for lexical retrieval."""
    tokens: list[str] = []
    for match in _LATIN_OR_CJK.finditer((text or "").lower()):
        value = match.group(0)
        if "\u3400" <= value[0] <= "\u9fff":
            tokens.extend(value)
            tokens.extend(value[i : i + 2] for i in range(len(value) - 1))
        else:
            tokens.append(value)
    return tokens


def split_text(text: str, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    """Split text into readable, overlapping chunks with deterministic bounds."""
    # Some non-compliant PDFs contain lone UTF-16 surrogate code points. Python strings can
    # temporarily hold them, but SQLite correctly rejects them when encoding to UTF-8.
    utf8_safe = (text or "").encode("utf-8", errors="replace").decode("utf-8")
    cleaned = re.sub(r"[ \t]+", " ", utf8_safe.replace("\r\n", "\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    overlap = max(0, min(overlap, max_chars // 3))

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        hard_end = min(len(cleaned), start + max_chars)
        end = hard_end
        if hard_end < len(cleaned):
            floor = start + max_chars // 2
            candidates = [
                cleaned.rfind("\n\n", floor, hard_end),
                cleaned.rfind("。", floor, hard_end),
                cleaned.rfind(". ", floor, hard_end),
                cleaned.rfind("\n", floor, hard_end),
            ]
            best = max(candidates)
            if best > floor:
                end = best + (1 if cleaned[best] != "." else 2)

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap)
    return chunks


class ResearchStore:
    """SQLite-backed source, evidence, memory, and audit store."""

    def __init__(self, workspace: Path, db_path: Path | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = ensure_dir(self.workspace / "research")
        self.db_path = db_path or self.root / "research.db"
        ensure_dir(self.db_path.parent)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    page INTEGER,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE,
                    UNIQUE(source_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_project
                    ON memories(project, status);
                CREATE TABLE IF NOT EXISTS research_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO research_events(event_type, payload_json, created_at) VALUES (?, ?, ?)",
                (event_type, json.dumps(payload, ensure_ascii=False), _now()),
            )

    @staticmethod
    def _read_pdf(path: Path) -> list[tuple[int | None, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError("PDF support requires pypdf; reinstall nanobot dependencies") from exc

        # pypdf can repair many malformed cross-reference tables in non-strict mode. Its warning
        # stream is extremely noisy for such documents, so keep the CLI concise while still
        # surfacing actual per-file failures in ingest_path().
        pypdf_logger = logging.getLogger("pypdf")
        previous_level = pypdf_logger.level
        pypdf_logger.setLevel(logging.ERROR)
        try:
            reader = PdfReader(str(path), strict=False)
            return [
                (index, (page.extract_text() or "").strip())
                for index, page in enumerate(reader.pages, start=1)
            ]
        finally:
            pypdf_logger.setLevel(previous_level)

    @staticmethod
    def _read_text(path: Path) -> list[tuple[int | None, str]]:
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        if path.suffix.lower() in {".html", ".htm"}:
            text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
        return [(None, text)]

    def ingest_file(
        self,
        path: Path,
        *,
        title: str | None = None,
        max_chars: int = 1200,
        overlap: int = 160,
    ) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported source type '{suffix or '(none)'}'")

        raw = path.read_bytes()
        digest = _sha256(raw)
        source_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, sha256 FROM sources WHERE path = ?", (str(path),)
            ).fetchone()
        if existing and existing["sha256"] == digest:
            chunk_count = self._source_chunk_count(existing["id"])
            return {
                "source_id": existing["id"],
                "path": str(path),
                "chunks": chunk_count,
                "status": "unchanged",
            }

        pages = self._read_pdf(path) if suffix == ".pdf" else self._read_text(path)
        chunk_rows: list[tuple[str, str, int, int | None, str, int]] = []
        ordinal = 1
        for page, text in pages:
            for content in split_text(text, max_chars=max_chars, overlap=overlap):
                citation = f"RF-{source_id[:8]}-{ordinal}"
                chunk_rows.append(
                    (citation, source_id, ordinal, page, content, len(tokenize(content)))
                )
                ordinal += 1
        if not chunk_rows:
            raise ValueError(f"No extractable text found in {path.name}")

        with self._connect() as conn:
            if existing:
                conn.execute("DELETE FROM sources WHERE id = ?", (existing["id"],))
            conn.execute(
                "INSERT INTO sources(id, path, title, kind, sha256, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, str(path), title or path.stem, suffix.lstrip("."), digest, _now()),
            )
            conn.executemany(
                "INSERT INTO chunks(id, source_id, ordinal, page, content, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                chunk_rows,
            )
        result = {
            "source_id": source_id,
            "path": str(path),
            "chunks": len(chunk_rows),
            "pages": len(pages) if suffix == ".pdf" else None,
            "status": "updated" if existing else "indexed",
        }
        self.log_event("source_ingested", result)
        return result

    def ingest_path(self, path: Path, *, max_files: int = 100) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        if path.is_file():
            result = self.ingest_file(path)
            return {"indexed": [result], "errors": [], "total": 1}
        if not path.is_dir():
            raise FileNotFoundError(f"Source path not found: {path}")

        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        )[:max_files]
        indexed: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for candidate in files:
            try:
                indexed.append(self.ingest_file(candidate))
            except Exception as exc:
                errors.append({"path": str(candidate), "error": str(exc)})
        return {"indexed": indexed, "errors": errors, "total": len(files)}

    def _source_chunk_count(self, source_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchone()
        return int(row["count"] if row else 0)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, COUNT(c.id) AS chunks
                FROM sources s LEFT JOIN chunks c ON c.source_id = s.id
                GROUP BY s.id ORDER BY s.indexed_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, source_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Return corpus chunks in stable citation order for alternate retrievers."""
        selected = [value for value in (source_ids or []) if value]
        params: list[Any] = []
        where = ""
        if selected:
            where = "WHERE c.source_id IN ({})".format(",".join("?" for _ in selected))
            params.extend(selected)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS citation, c.source_id, c.ordinal, c.page, c.content,
                       s.title, s.path
                FROM chunks c JOIN sources s ON s.id = c.source_id
                {where}
                ORDER BY c.id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        source_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        params: list[Any] = []
        where = ""
        selected = [value for value in (source_ids or []) if value]
        if selected:
            where = "WHERE c.source_id IN ({})".format(",".join("?" for _ in selected))
            params.extend(selected)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.source_id, c.ordinal, c.page, c.content,
                       s.title, s.path
                FROM chunks c JOIN sources s ON s.id = c.source_id
                {where}
                """,
                params,
            ).fetchall()
        if not rows:
            return []

        docs = [tokenize(row["content"]) for row in rows]
        avg_len = sum(len(doc) for doc in docs) / max(1, len(docs))
        doc_freq = Counter()
        for doc in docs:
            doc_freq.update(set(doc))

        query_counts = Counter(query_tokens)
        scored: list[tuple[float, sqlite3.Row]] = []
        n_docs = len(docs)
        k1, b = 1.5, 0.75
        for row, tokens in zip(rows, docs):
            counts = Counter(tokens)
            score = 0.0
            for term, query_weight in query_counts.items():
                freq = counts.get(term, 0)
                if not freq:
                    continue
                df = doc_freq[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * len(tokens) / max(1.0, avg_len))
                score += idf * (freq * (k1 + 1) / denom) * min(query_weight, 2)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        results = []
        for score, row in scored[: max(1, min(top_k, 20))]:
            content = row["content"]
            results.append(
                {
                    "citation": row["id"],
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "path": row["path"],
                    "page": row["page"],
                    "score": round(score, 4),
                    "content": content,
                }
            )
        self.log_event(
            "evidence_searched",
            {"query": query, "top_k": top_k, "hits": [item["citation"] for item in results]},
        )
        return results

    def get_chunks(self, citations: Iterable[str]) -> list[dict[str, Any]]:
        normalized = [value.upper() for value in citations if value]
        if not normalized:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id AS citation, c.source_id, c.ordinal, c.page, c.content,
                       s.title, s.path
                FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE UPPER(c.id) IN ({})
                """.format(",".join("?" for _ in normalized)),
                normalized,
            ).fetchall()
        by_id = {row["citation"].upper(): dict(row) for row in rows}
        return [by_id[value] for value in normalized if value in by_id]

    def remember(
        self,
        project: str,
        kind: str,
        content: str,
        citations: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        project = project.strip() or "default"
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        citations_list = [value.upper() for value in (citations or [])]
        missing = self.missing_citations(citations_list)
        if missing:
            raise ValueError(f"Unknown citations: {', '.join(missing)}")
        memory_id = "RM-" + hashlib.sha1(
            f"{project}\0{kind}\0{content}".encode("utf-8")
        ).hexdigest()[:12]
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(id, project, kind, content, citations_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET citations_json=excluded.citations_json,
                    status='active', updated_at=excluded.updated_at
                """,
                (memory_id, project, kind, content, json.dumps(citations_list), now, now),
            )
        result = {
            "id": memory_id,
            "project": project,
            "kind": kind,
            "content": content,
            "citations": citations_list,
        }
        self.log_event("memory_saved", result)
        return result

    def search_memories(self, project: str, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE project = ? AND status = 'active' ORDER BY updated_at DESC",
                (project.strip() or "default",),
            ).fetchall()
        query_tokens = set(tokenize(query))
        results = []
        for row in rows:
            data = dict(row)
            data["citations"] = json.loads(data.pop("citations_json"))
            score = len(query_tokens.intersection(tokenize(data["content"]))) if query_tokens else 1
            if score:
                data["score"] = score
                results.append(data)
        results.sort(key=lambda item: item["updated_at"], reverse=True)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[: max(1, min(limit, 50))]

    def supersede_memory(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
                (_now(), memory_id),
            )
        changed = cursor.rowcount > 0
        if changed:
            self.log_event("memory_superseded", {"id": memory_id})
        return changed

    def missing_citations(self, citations: Iterable[str]) -> list[str]:
        normalized = [value.upper() for value in citations if value]
        if not normalized:
            return []
        found = {row["citation"].upper() for row in self.get_chunks(normalized)}
        return [value for value in normalized if value not in found]

    def verify_report(self, content: str) -> dict[str, Any]:
        citations = [match.upper() for match in _CITATION.findall(content or "")]
        unique = list(dict.fromkeys(citations))
        missing = self.missing_citations(unique)
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", content or "")
            if len(paragraph.strip()) >= 60 and not paragraph.lstrip().startswith(("#", "```"))
        ]
        cited_paragraphs = sum(bool(_CITATION.search(paragraph)) for paragraph in paragraphs)
        coverage = cited_paragraphs / len(paragraphs) if paragraphs else 1.0
        return {
            "valid": not missing and bool(unique),
            "citations": unique,
            "missing": missing,
            "claim_paragraphs": len(paragraphs),
            "cited_paragraphs": cited_paragraphs,
            "citation_coverage": round(coverage, 4),
        }

    def save_report(self, title: str, content: str) -> dict[str, Any]:
        verification = self.verify_report(content)
        if not verification["valid"]:
            details = ", ".join(verification["missing"]) or "no evidence citations found"
            raise ValueError(f"Report citation verification failed: {details}")
        reports_dir = ensure_dir(self.root / "reports")
        base = safe_filename(title).strip(" .") or "research-report"
        report_path = reports_dir / f"{base}.md"
        suffix = 2
        while report_path.exists():
            report_path = reports_dir / f"{base}-{suffix}.md"
            suffix += 1
        report_path.write_text(content, encoding="utf-8")
        result = {"path": str(report_path), "verification": verification}
        self.log_event("report_saved", result)
        return result

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            sources = conn.execute("SELECT COUNT(*) AS n FROM sources").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            memories = conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE status = 'active'"
            ).fetchone()["n"]
            events = conn.execute("SELECT COUNT(*) AS n FROM research_events").fetchone()["n"]
        return {
            "database": str(self.db_path),
            "sources": sources,
            "chunks": chunks,
            "active_memories": memories,
            "events": events,
        }
