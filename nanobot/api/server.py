"""FastAPI HTTP surface for nanobot and the ResearchPilot application."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.datastructures import UploadFile

from nanobot.api.research import TERMINAL_STATUSES, ResearchTaskManager
from nanobot.research.store import SUPPORTED_SUFFIXES
from nanobot.utils.helpers import safe_filename

API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "code": status}}, status_code=status
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)


def _research_manager(request: Request) -> ResearchTaskManager | None:
    return getattr(request.app.state, "research_manager", None)


async def handle_chat_completions(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return _error_json(400, "Invalid JSON body")
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return _error_json(400, "Only a single user message is supported")
    if body.get("stream", False):
        return _error_json(400, "stream=true is not supported yet. Set stream=false or omit it.")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        return _error_json(400, "Only a single user message is supported")
    user_content = message.get("content", "")
    if isinstance(user_content, list):
        user_content = " ".join(
            str(part.get("text", ""))
            for part in user_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    agent_loop = request.app.state.agent_loop
    timeout_s = float(request.app.state.request_timeout)
    model_name = str(request.app.state.model_name)
    if (requested_model := body.get("model")) and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")
    session_key = f"api:{body['session_id']}" if body.get("session_id") else API_SESSION_KEY
    locks: dict[str, asyncio.Lock] = request.app.state.session_locks
    session_lock = locks.setdefault(session_key, asyncio.Lock())
    fallback = "I've completed processing but have no response to give."
    logger.info("API request session_key={} content={}", session_key, str(user_content)[:80])
    try:
        async with session_lock:
            response = await asyncio.wait_for(
                agent_loop.process_direct(
                    content=user_content, session_key=session_key, channel="api", chat_id=API_CHAT_ID
                ),
                timeout=timeout_s,
            )
            response_text = _response_text(response)
            if not response_text.strip():
                logger.warning("Empty response for session {}, retrying", session_key)
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=user_content,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response) or fallback
    except asyncio.TimeoutError:
        return _error_json(504, f"Request timed out after {timeout_s}s")
    except Exception:
        logger.exception("Error processing request for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")
    return JSONResponse(_chat_completion_response(response_text, model_name))


async def handle_models(request: Request) -> JSONResponse:
    model_name = str(request.app.state.model_name)
    return JSONResponse({"object": "list", "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "nanobot"}]})


async def handle_health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "research": "configured" if _research_manager(request) is not None else "disabled",
            "retrieval_backend": os.getenv("RESEARCH_RETRIEVAL_BACKEND", "local"),
        }
    )


async def handle_index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


async def handle_upload_document(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    if not request.headers.get("content-type", "").startswith("multipart/"):
        return _error_json(415, "Expected multipart/form-data with a file field")
    try:
        form = await request.form()
    except Exception:
        return _error_json(400, "Invalid multipart form")
    field = form.get("file")
    if not isinstance(field, UploadFile) or not field.filename:
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
            while chunk := await field.read(64 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("Document exceeds the 25 MB upload limit")
                output.write(chunk)
        result = await asyncio.to_thread(manager.research.ingest_file, target, title=Path(filename).stem)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return _error_json(413 if "25 MB" in str(exc) else 400, str(exc))
    finally:
        await field.close()
    return JSONResponse({"document": result}, status_code=201)


async def handle_list_documents(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    return JSONResponse({"data": await asyncio.to_thread(manager.research.list_sources)})


async def handle_create_research_task(request: Request) -> JSONResponse:
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
    record = manager.submit(
        query,
        title=str(body.get("title", "")).strip() or None,
        project=str(body.get("project", "default")).strip() or "default",
    )
    return JSONResponse({"task": record}, status_code=202)


async def handle_list_research_tasks(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        return _error_json(400, "limit must be an integer")
    return JSONResponse({"data": manager.store.list(limit=limit)})


async def handle_get_research_task(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    record = manager.store.get(request.path_params["task_id"])
    if not record:
        return _error_json(404, "Research task not found")
    return JSONResponse({"task": record})


async def handle_cancel_research_task(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    record = manager.cancel(request.path_params["task_id"])
    if not record:
        return _error_json(404, "Research task not found")
    return JSONResponse({"task": record}, status_code=202)


def _encode_sse(event: dict[str, Any]) -> bytes:
    payload = json.dumps(
        {"type": event["type"], "data": event["data"], "created_at": event["created_at"]},
        ensure_ascii=False,
    )
    return f"id: {event['id']}\nevent: {event['type']}\ndata: {payload}\n\n".encode()


async def handle_research_task_events(request: Request) -> Any:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    task_id = request.path_params["task_id"]
    if not manager.store.get(task_id):
        return _error_json(404, "Research task not found")
    raw_cursor = request.headers.get("Last-Event-ID", request.query_params.get("after", "0"))
    try:
        initial_cursor = max(0, int(raw_cursor))
    except ValueError:
        return _error_json(400, "Last-Event-ID must be an integer")

    async def stream() -> AsyncIterator[bytes]:
        cursor = initial_cursor
        while not await request.is_disconnected():
            events = manager.store.events_after(task_id, cursor)
            for event in events:
                yield _encode_sse(event)
                cursor = int(event["id"])
            record = manager.store.get(task_id)
            if record and record["status"] in TERMINAL_STATUSES and not events:
                break
            if not events:
                yield b": keep-alive\n\n"
                await manager.wait_for_events(task_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def handle_get_source(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    rows = await asyncio.to_thread(manager.research.get_chunks, [request.path_params["citation"].upper()])
    if not rows:
        return _error_json(404, "Citation not found")
    return JSONResponse({"source": rows[0]})


def _report_path(manager: ResearchTaskManager, name: str) -> Path | None:
    safe_name = safe_filename(Path(name).name).strip(" .")
    if not safe_name or safe_name != name or not safe_name.endswith(".md"):
        return None
    reports = (manager.workspace / "research" / "reports").resolve()
    path = (reports / safe_name).resolve()
    return path if path.parent == reports else None


async def handle_list_reports(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    reports_dir = manager.workspace / "research" / "reports"
    data = []
    if reports_dir.is_dir():
        for path in sorted(reports_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
            data.append({"name": path.name, "size": path.stat().st_size})
    return JSONResponse({"data": data})


async def handle_get_report(request: Request) -> JSONResponse:
    manager = _research_manager(request)
    if manager is None:
        return _error_json(503, "Research API is not configured")
    path = _report_path(manager, request.path_params["name"])
    if path is None or not path.is_file():
        return _error_json(404, "Report not found")
    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    verification = await asyncio.to_thread(manager.research.verify_report, content)
    return JSONResponse({"report": {"name": path.name, "content": content, "verification": verification}})


def create_app(
    agent_loop: Any,
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    workspace: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI application while preserving the existing API contract."""
    resolved_workspace = workspace
    if resolved_workspace is None:
        candidate = getattr(agent_loop, "workspace", None)
        if isinstance(candidate, (str, Path)):
            resolved_workspace = candidate
    research_manager = (
        ResearchTaskManager(agent_loop, Path(resolved_workspace))
        if resolved_workspace is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        connect = getattr(agent_loop, "_connect_mcp", None)
        if connect is not None:
            await connect()
        try:
            yield
        finally:
            if research_manager is not None:
                await research_manager.close()
            close_mcp = getattr(agent_loop, "close_mcp", None)
            if close_mcp is not None:
                await close_mcp()

    app = FastAPI(title="ResearchPilot API", version="1.0.0", lifespan=lifespan)
    app.state.agent_loop = agent_loop
    app.state.model_name = model_name
    app.state.request_timeout = request_timeout
    app.state.session_locks = {}
    app.state.research_manager = research_manager
    app.add_api_route("/v1/chat/completions", handle_chat_completions, methods=["POST"])
    app.add_api_route("/v1/models", handle_models, methods=["GET"])
    app.add_api_route("/health", handle_health, methods=["GET"])
    app.add_api_route("/", handle_index, methods=["GET"])
    app.add_api_route("/api/v1/documents", handle_upload_document, methods=["POST"])
    app.add_api_route("/api/v1/documents", handle_list_documents, methods=["GET"])
    app.add_api_route("/api/v1/research/tasks", handle_create_research_task, methods=["POST"])
    app.add_api_route("/api/v1/research/tasks", handle_list_research_tasks, methods=["GET"])
    app.add_api_route("/api/v1/research/tasks/{task_id}", handle_get_research_task, methods=["GET"])
    app.add_api_route("/api/v1/research/tasks/{task_id}/cancel", handle_cancel_research_task, methods=["POST"])
    app.add_api_route("/api/v1/research/tasks/{task_id}/events", handle_research_task_events, methods=["GET"])
    app.add_api_route("/api/v1/sources/{citation}", handle_get_source, methods=["GET"])
    app.add_api_route("/api/v1/reports", handle_list_reports, methods=["GET"])
    app.add_api_route("/api/v1/reports/{name}", handle_get_report, methods=["GET"])
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="researchflow-assets")
    return app
