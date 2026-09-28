"""Integration tests for the asynchronous ResearchFlow API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import pytest_asyncio

from nanobot.api.research import ResearchTaskManager, ResearchTaskStore
from nanobot.api.server import create_app


class _Response:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status = response.status_code

    async def json(self):
        return self._response.json()

    async def text(self):
        return self._response.text


class _Client:
    def __init__(self, app) -> None:
        self.app = app
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def post(self, *args, **kwargs):
        return _Response(await self._client.post(*args, **kwargs))

    async def get(self, *args, **kwargs):
        return _Response(await self._client.get(*args, **kwargs))

    async def close(self):
        await self._client.aclose()


@pytest_asyncio.fixture
async def api_client():
    clients: list[_Client] = []

    async def factory(app):
        client = _Client(app)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        await client.close()


def _agent(workspace: Path, process) -> MagicMock:
    agent = MagicMock()
    agent.workspace = workspace
    agent.process_direct = process
    return agent


async def _wait_for_terminal(client: _Client, task_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/api/v1/research/tasks/{task_id}")
        task = (await response.json())["task"]
        if task["status"] in {"succeeded", "failed", "cancelled"}:
            return task
        await asyncio.sleep(0.01)
    raise AssertionError("task did not complete")


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    """Wait for an in-process task assertion without adding fixed sleeps."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_upload_and_read_document(api_client, tmp_path: Path) -> None:
    async def process(**_):
        return SimpleNamespace(content="done")

    client = await api_client(create_app(_agent(tmp_path, process), workspace=tmp_path))
    response = await client.post(
        "/api/v1/documents",
        files={
            "file": (
                "sample.txt",
                b"HyperLogLog estimates distinct cardinality with compact registers.",
                "text/plain",
            )
        },
    )
    assert response.status == 201
    document = (await response.json())["document"]
    assert document["status"] == "indexed"
    assert document["chunks"] == 1

    response = await client.get("/api/v1/documents")
    sources = (await response.json())["data"]
    assert len(sources) == 1

    citation = f"RF-{document['source_id'][:8]}-1"
    response = await client.get(f"/api/v1/sources/{citation}")
    assert response.status == 200
    assert (await response.json())["source"]["citation"] == citation


@pytest.mark.asyncio
async def test_task_lifecycle_and_sse_replay(api_client, tmp_path: Path) -> None:
    async def process(**kwargs):
        await kwargs["on_progress"]("research_search(\"HLL\")", tool_hint=True)
        await kwargs["on_stream"]("grounded ")
        await kwargs["on_stream"]("answer")
        await kwargs["on_stream_end"](resuming=False)
        return SimpleNamespace(content="grounded answer [RF-12345678-1]")

    client = await api_client(create_app(_agent(tmp_path, process), workspace=tmp_path))
    response = await client.post(
        "/api/v1/research/tasks",
        json={"query": "Compare cardinality estimators", "project": "benchmark"},
    )
    assert response.status == 202
    task_id = (await response.json())["task"]["id"]
    task = await _wait_for_terminal(client, task_id)
    assert task["status"] == "succeeded"
    assert task["result"].startswith("grounded answer")

    response = await client.get(f"/api/v1/research/tasks/{task_id}/events")
    assert response.status == 200
    body = await response.text()
    assert "event: task.created" in body
    assert "event: agent.progress" in body
    assert body.count("event: agent.delta") == 2
    assert "event: task.completed" in body


@pytest.mark.asyncio
async def test_task_can_be_cancelled(api_client, tmp_path: Path) -> None:
    started = asyncio.Event()

    async def process(**_):
        started.set()
        await asyncio.Event().wait()

    client = await api_client(create_app(_agent(tmp_path, process), workspace=tmp_path))
    response = await client.post("/api/v1/research/tasks", json={"query": "long task"})
    task_id = (await response.json())["task"]["id"]
    await asyncio.wait_for(started.wait(), timeout=1)

    response = await client.post(f"/api/v1/research/tasks/{task_id}/cancel")
    assert response.status == 202
    task = await _wait_for_terminal(client, task_id)
    assert task["status"] == "cancelled"


@pytest.mark.asyncio
async def test_report_listing_and_verification(api_client, tmp_path: Path) -> None:
    async def process(**_):
        return SimpleNamespace(content="done")

    client = await api_client(create_app(_agent(tmp_path, process), workspace=tmp_path))
    manager = client.app.state.research_manager
    source = tmp_path / "source.txt"
    source.write_text("evidence for report", encoding="utf-8")
    indexed = manager.research.ingest_file(source)
    citation = f"RF-{indexed['source_id'][:8]}-1"
    manager.research.save_report("demo", f"# Demo\n\nA grounded statement [{citation}].")

    response = await client.get("/api/v1/reports")
    assert (await response.json())["data"][0]["name"] == "demo.md"
    response = await client.get("/api/v1/reports/demo.md")
    report = (await response.json())["report"]
    assert report["verification"]["valid"] is True


@pytest.mark.asyncio
async def test_research_routes_require_workspace(api_client) -> None:
    agent = MagicMock(spec=[])
    client = await api_client(create_app(agent))
    response = await client.get("/api/v1/research/tasks")
    assert response.status == 503


@pytest.mark.asyncio
async def test_web_client_is_served(api_client) -> None:
    agent = MagicMock(spec=[])
    client = await api_client(create_app(agent))
    response = await client.get("/")
    assert response.status == 200
    assert "ResearchFlow" in await response.text()
    response = await client.get("/assets/app.js")
    assert response.status == 200
    assert "EventSource" in await response.text()


@pytest.mark.asyncio
async def test_task_manager_bounds_concurrency_and_isolates_results(tmp_path: Path) -> None:
    """Only two agents run together; queued work and task results stay isolated."""
    release = asyncio.Event()
    active = 0
    max_active = 0
    started_ids: set[str] = set()

    async def process(**kwargs):
        nonlocal active, max_active
        task_id = kwargs["chat_id"]
        active += 1
        max_active = max(max_active, active)
        started_ids.add(task_id)
        try:
            await release.wait()
            return SimpleNamespace(content=f"result:{task_id}")
        finally:
            active -= 1

    manager = ResearchTaskManager(
        _agent(tmp_path, process),
        tmp_path,
        max_concurrency=2,
    )
    records = [manager.submit(f"query-{index}") for index in range(3)]
    task_ids = [record["id"] for record in records]

    try:
        await _wait_until(lambda: len(started_ids) == 2)
        statuses = [manager.store.get(task_id)["status"] for task_id in task_ids]
        assert statuses.count("running") == 2
        assert statuses.count("pending") == 1
        assert max_active == 2

        release.set()
        await _wait_until(
            lambda: all(
                manager.store.get(task_id)["status"] == "succeeded"
                for task_id in task_ids
            )
        )
        for task_id in task_ids:
            task = manager.store.get(task_id)
            assert task["result"] == f"result:{task_id}"
        assert max_active == 2
    finally:
        release.set()
        await manager.close()


def test_task_store_recovers_interrupted_work_after_restart(tmp_path: Path) -> None:
    """A new store instance fails orphaned work and remains ready for new tasks."""
    original = ResearchTaskStore(tmp_path)
    pending = original.create("queued query", title=None, project="recovery")
    running = original.create("active query", title=None, project="recovery")
    original.update(running["id"], "running", started_at="2026-01-01T00:00:00+00:00")

    restarted = ResearchTaskStore(tmp_path)
    for task_id in (pending["id"], running["id"]):
        recovered = restarted.get(task_id)
        assert recovered["status"] == "failed"
        assert recovered["error"] == "Service restarted while task was running"
        assert recovered["completed_at"] is not None

    new_task = restarted.create("new query", title=None, project="recovery")
    assert new_task["status"] == "pending"
    assert new_task["error"] is None
