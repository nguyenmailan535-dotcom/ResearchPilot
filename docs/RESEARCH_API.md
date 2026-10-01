# ResearchPilot HTTP API

Start the service with the same nanobot configuration used by the CLI:

```powershell
nanobot serve --host 127.0.0.1 --port 18791 --workspace .\research-demo
```

The API persists task state and replayable events in the workspace SQLite database. At most two
research tasks run concurrently; additional tasks remain pending. A service restart marks
interrupted tasks as failed instead of leaving them permanently in `running` state.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/documents` | Upload and index one multipart document (`file`, max 25 MB) |
| `GET` | `/api/v1/documents` | List indexed sources |
| `POST` | `/api/v1/research/tasks` | Submit an asynchronous research task |
| `GET` | `/api/v1/research/tasks` | List recent tasks |
| `GET` | `/api/v1/research/tasks/{id}` | Read durable task state and result |
| `POST` | `/api/v1/research/tasks/{id}/cancel` | Cancel a pending or running task |
| `GET` | `/api/v1/research/tasks/{id}/events` | Replay and follow task events over SSE |
| `GET` | `/api/v1/sources/{citation}` | Read an exact evidence chunk |
| `GET` | `/api/v1/reports` | List generated Markdown reports |
| `GET` | `/api/v1/reports/{name}` | Read a report and its citation verification |

Create a task:

```powershell
$body = @{
  query = "Compare the indexed cardinality estimators and recommend one for distributed use"
  project = "sketch-comparison"
} | ConvertTo-Json

$task = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:18791/api/v1/research/tasks `
  -ContentType application/json -Body $body
```

SSE is resumable. Reconnect with the last received event ID in either the `Last-Event-ID` header
or the `?after=<id>` query parameter. Persisted event types are `task.created`, `task.status`,
`agent.progress`, `agent.delta`, `agent.stream_end`, and `task.completed`.

## Docker

Copy `.env.example` to `.env`, keep the provider credentials in the mounted nanobot
`config.json`, and run:

```bash
docker compose -f docker-compose.research.yml up --build
```

The UI and API are then available at `http://localhost:18791`.
