"""Persistent asynchronous task service for the ResearchFlow HTTP API."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from nanobot.research.store import ResearchStore

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


class ResearchTaskStore:
    """SQLite persistence for task state and replayable task events."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.db_path = self.workspace / "research" / "research.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
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
                CREATE TABLE IF NOT EXISTS api_research_tasks (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    title TEXT,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_api_research_tasks_created
                    ON api_research_tasks(created_at DESC);
                CREATE TABLE IF NOT EXISTS api_research_task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES api_research_tasks(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_api_research_task_events
                    ON api_research_task_events(task_id, id);
                """
            )
            now = _now()
            conn.execute(
                """
                UPDATE api_research_tasks
                SET status='failed', error='Service restarted while task was running',
                    completed_at=?, updated_at=?
                WHERE status IN ('pending', 'running')
                """,
                (now, now),
            )

    def create(self, query: str, *, title: str | None, project: str) -> dict[str, Any]:
        task_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_research_tasks(
                    id, query, title, project, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                """,
                (task_id, query, title, project, now, now),
            )
        self.append_event(task_id, "task.created", {"status": "pending"})
        return self.get(task_id) or {}

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM api_research_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row else None

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_research_tasks ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def update(self, task_id: str, status: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"result", "error", "started_at", "completed_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = status
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE api_research_tasks SET {assignments} WHERE id = ?",
                (*updates.values(), task_id),
            )
        return self.get(task_id)

    def append_event(self, task_id: str, event_type: str, data: dict[str, Any]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO api_research_task_events(task_id, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, event_type, json.dumps(data, ensure_ascii=False), _now()),
            )
            return int(cursor.lastrowid)

    def events_after(self, task_id: str, event_id: int = 0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, event_type, data_json, created_at
                FROM api_research_task_events
                WHERE task_id = ? AND id > ? ORDER BY id
                """,
                (task_id, event_id),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "type": row["event_type"],
                "data": json.loads(row["data_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


class ResearchTaskManager:
    """Run ResearchFlow requests asynchronously with bounded concurrency."""

    def __init__(self, agent_loop: Any, workspace: Path, *, max_concurrency: int = 2) -> None:
        self.agent_loop = agent_loop
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = ResearchTaskStore(self.workspace)
        self.research = ResearchStore(self.workspace)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._signals: dict[str, asyncio.Event] = {}

    def submit(self, query: str, *, title: str | None = None, project: str = "default") -> dict[str, Any]:
        record = self.store.create(query, title=title, project=project)
        task_id = record["id"]
        task = asyncio.create_task(self._run(task_id), name=f"research:{task_id}")
        self._tasks[task_id] = task
        self._signals[task_id] = asyncio.Event()
        task.add_done_callback(lambda _: self._tasks.pop(task_id, None))
        return record

    def _notify(self, task_id: str) -> None:
        signal = self._signals.get(task_id)
        if signal:
            signal.set()

    def emit(self, task_id: str, event_type: str, data: dict[str, Any]) -> None:
        self.store.append_event(task_id, event_type, data)
        self._notify(task_id)

    async def _run(self, task_id: str) -> None:
        record = self.store.get(task_id)
        if not record:
            return
        try:
            async with self._semaphore:
                started = _now()
                self.store.update(task_id, "running", started_at=started)
                self.emit(task_id, "task.status", {"status": "running"})

                async def on_progress(content: str, *, tool_hint: bool = False) -> None:
                    self.emit(
                        task_id,
                        "agent.progress",
                        {"content": content, "tool_hint": tool_hint},
                    )

                async def on_stream(content: str) -> None:
                    if content:
                        self.emit(task_id, "agent.delta", {"content": content})

                async def on_stream_end(*, resuming: bool = False) -> None:
                    self.emit(task_id, "agent.stream_end", {"resuming": resuming})

                prompt = (
                    "Use the research-flow skill. Complete the following research task using "
                    "only evidence from the indexed corpus. Verify every RF citation and save a "
                    "Markdown report with research_report before finishing.\n\n"
                    f"Project: {record['project']}\nTask: {record['query']}"
                )
                response = await self.agent_loop.process_direct(
                    content=prompt,
                    session_key=f"research-api:{task_id}",
                    channel="research-api",
                    chat_id=task_id,
                    on_progress=on_progress,
                    on_stream=on_stream,
                    on_stream_end=on_stream_end,
                )
                result = "" if response is None else str(getattr(response, "content", response) or "")
                completed = _now()
                self.store.update(
                    task_id,
                    "succeeded",
                    result=result,
                    completed_at=completed,
                )
                self.emit(task_id, "task.completed", {"status": "succeeded", "result": result})
        except asyncio.CancelledError:
            completed = _now()
            self.store.update(task_id, "cancelled", completed_at=completed)
            self.emit(task_id, "task.completed", {"status": "cancelled"})
        except Exception as exc:
            completed = _now()
            self.store.update(task_id, "failed", error=str(exc), completed_at=completed)
            self.emit(task_id, "task.completed", {"status": "failed", "error": str(exc)})

    def cancel(self, task_id: str) -> dict[str, Any] | None:
        record = self.store.get(task_id)
        if not record:
            return None
        if record["status"] in TERMINAL_STATUSES:
            return record
        task = self._tasks.get(task_id)
        if task:
            task.cancel()
        else:
            self.store.update(task_id, "cancelled", completed_at=_now())
            self.emit(task_id, "task.completed", {"status": "cancelled"})
        return self.store.get(task_id)

    async def wait_for_events(self, task_id: str, *, timeout: float = 15.0) -> None:
        signal = self._signals.setdefault(task_id, asyncio.Event())
        signal.clear()
        try:
            await asyncio.wait_for(signal.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
