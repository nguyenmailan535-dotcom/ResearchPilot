"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web
from loguru import logger

from nanobot.api.research import TERMINAL_STATUSES, ResearchTaskManager
from nanobot.research.store import SUPPORTED_SUFFIXES
from nanobot.utils.helpers import safe_filename

API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions"""

    # --- Parse body ---
    try:
        body = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")

    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return _error_json(400, "Only a single user message is supported")

    # Stream not yet supported
    if body.get("stream", False):
        return _error_json(400, "stream=true is not supported yet. Set stream=false or omit it.")

    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return _error_json(400, "Only a single user message is supported")
    user_content = message.get("content", "")
    if isinstance(user_content, list):
        # Multi-modal content array — extract text parts
        user_content = " ".join(
            part.get("text", "") for part in user_content if part.get("type") == "text"
        )

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "nanobot")
    if (requested_model := body.get("model")) and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{body['session_id']}" if body.get("session_id") else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info("API request session_key={} content={}", session_key, user_content[:80])

    fallback = "I've completed processing but have no response to give."

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=user_content,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning(
                        "Empty response for session {}, retrying",
                        session_key,
                    )
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=user_content,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning(
                            "Empty response after retry for session {}, using fallback",
                            session_key,
                        )
                        response_text = fallback

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(_chat_completion_response(response_text, model_name))


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = request.app.get("model_name", "nanobot")
    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "nanobot",
            }
        ],
    })


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


async def handle_index(request: web.Request) -> web.FileResponse:
    """Serve the zero-build ResearchFlow web client."""
    return web.FileResponse(WEB_ROOT / "index.html")


def _research_manager(request: web.Request) -> ResearchTaskManager | None:
    return request.app.get("research_manager")


async def handle_upload_document(request: web.Request) -> web.Response:
    """POST /api/v1/documents with one multipart ``file`` field."""
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    if not request.content_type.startswith("multipart/"):
        return _error_json(415, "Expected multipart/form-data with a file field")

    reader = await request.multipart()
    field = await reader.next()
    while field is not None and field.name != "file":
        field = await reader.next()
    if field is None or not field.filename:
        return _error_json(400, "Missing multipart file field")

    filename = safe_filename(Path(field.filename).name).strip(" .")
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_SUFFIXES:
        return _error_json(400, f"Unsupported document type '{suffix or '(none)'}'")

    upload_dir = manager.workspace / "research" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex[:12]}-{filename}"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := await field.read_chunk(size=64 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("Document exceeds the 25 MB upload limit")
                output.write(chunk)
        result = await asyncio.to_thread(manager.research.ingest_file, target, title=Path(filename).stem)
    except Exception as exc:
        target.unlink(missing_ok=True)
        status = 413 if "25 MB" in str(exc) else 400
        return _error_json(status, str(exc))
    return web.json_response({"document": result}, status=201)


async def handle_list_documents(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    sources = await asyncio.to_thread(manager.research.list_sources)
    return web.json_response({"data": sources})


async def handle_create_research_task(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    try:
        body = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")
    query = str(body.get("query", "")).strip()
    if not query:
        return _error_json(400, "query is required")
    if len(query) > 10_000:
        return _error_json(400, "query must not exceed 10000 characters")
    title = str(body.get("title", "")).strip() or None
    project = str(body.get("project", "default")).strip() or "default"
    record = manager.submit(query, title=title, project=project)
    return web.json_response({"task": record}, status=202)


async def handle_list_research_tasks(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        return _error_json(400, "limit must be an integer")
    return web.json_response({"data": manager.store.list(limit=limit)})


async def handle_get_research_task(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    record = manager.store.get(request.match_info["task_id"])
    if not record:
        return _error_json(404, "Research task not found")
    return web.json_response({"task": record})


async def handle_cancel_research_task(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    record = manager.cancel(request.match_info["task_id"])
    if not record:
        return _error_json(404, "Research task not found")
    return web.json_response({"task": record}, status=202)


def _encode_sse(event: dict[str, Any]) -> bytes:
    payload = json.dumps(
        {"type": event["type"], "data": event["data"], "created_at": event["created_at"]},
        ensure_ascii=False,
    )
    return f"id: {event['id']}\nevent: {event['type']}\ndata: {payload}\n\n".encode()


async def handle_research_task_events(request: web.Request) -> web.StreamResponse:
    """Replay persisted events and follow new events using Server-Sent Events."""
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    task_id = request.match_info["task_id"]
    if not manager.store.get(task_id):
        return _error_json(404, "Research task not found")
    raw_cursor = request.headers.get("Last-Event-ID", request.query.get("after", "0"))
    try:
        cursor = max(0, int(raw_cursor))
    except ValueError:
        return _error_json(400, "Last-Event-ID must be an integer")

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    try:
        while True:
            events = manager.store.events_after(task_id, cursor)
            for event in events:
                await response.write(_encode_sse(event))
                cursor = event["id"]
            record = manager.store.get(task_id)
            if record and record["status"] in TERMINAL_STATUSES and not events:
                break
            if not events:
                await response.write(b": keep-alive\n\n")
                await manager.wait_for_events(task_id)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def handle_get_source(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    citation = request.match_info["citation"].upper()
    rows = await asyncio.to_thread(manager.research.get_chunks, [citation])
    if not rows:
        return _error_json(404, "Citation not found")
    return web.json_response({"source": rows[0]})


def _report_path(manager: ResearchTaskManager, name: str) -> Path | None:
    safe_name = safe_filename(Path(name).name).strip(" .")
    if not safe_name or safe_name != name or not safe_name.endswith(".md"):
        return None
    reports = (manager.workspace / "research" / "reports").resolve()
    path = (reports / safe_name).resolve()
    return path if path.parent == reports else None


async def handle_list_reports(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    reports_dir = manager.workspace / "research" / "reports"
    data = []
    if reports_dir.is_dir():
        for path in sorted(reports_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
            data.append({"name": path.name, "size": path.stat().st_size})
    return web.json_response({"data": data})


async def handle_get_report(request: web.Request) -> web.Response:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    path = _report_path(manager, request.match_info["name"])
    if path is None or not path.is_file():
        return _error_json(404, "Report not found")
    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    verification = await asyncio.to_thread(manager.research.verify_report, content)
    return web.json_response({"report": {"name": path.name, "content": content, "verification": verification}})


async def _close_research_manager(app: web.Application) -> None:
    manager = app.get("research_manager")
    if manager is not None:
        await manager.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    agent_loop,
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    workspace: str | Path | None = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES + 1024 * 1024)
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key

    resolved_workspace = workspace
    if resolved_workspace is None:
        candidate = getattr(agent_loop, "workspace", None)
        if isinstance(candidate, (str, Path)):
            resolved_workspace = candidate
    app["research_manager"] = (
        ResearchTaskManager(agent_loop, Path(resolved_workspace))
        if resolved_workspace is not None
        else None
    )
    app.on_cleanup.append(_close_research_manager)

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/v1/documents", handle_upload_document)
    app.router.add_get("/api/v1/documents", handle_list_documents)
    app.router.add_post("/api/v1/research/tasks", handle_create_research_task)
    app.router.add_get("/api/v1/research/tasks", handle_list_research_tasks)
    app.router.add_get("/api/v1/research/tasks/{task_id}", handle_get_research_task)
    app.router.add_post("/api/v1/research/tasks/{task_id}/cancel", handle_cancel_research_task)
    app.router.add_get("/api/v1/research/tasks/{task_id}/events", handle_research_task_events)
    app.router.add_get("/api/v1/sources/{citation}", handle_get_source)
    app.router.add_get("/api/v1/reports", handle_list_reports)
    app.router.add_get("/api/v1/reports/{name}", handle_get_report)
    app.router.add_static("/assets/", WEB_ROOT, name="researchflow-assets")
    return app
