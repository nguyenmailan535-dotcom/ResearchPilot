const state = { tasks: [], selected: null, stream: null, events: 0, streamed: "", stage: -1 };
const $ = (id) => document.getElementById(id);

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || `HTTP ${response.status}`);
  return payload;
}

function shortDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function taskLabel(task) { return task.title || task.query || task.id; }

function appendInline(parent, text) {
  const pattern = /(\[RF-[a-f0-9]{8}-\d+\]|\*\*[^*]+\*\*|`[^`]+`)/gi;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (/^\[RF-/i.test(token)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "citation-link";
      button.dataset.citation = token.slice(1, -1).toUpperCase();
      button.textContent = token;
      button.setAttribute("aria-label", `查看证据 ${button.dataset.citation}`);
      parent.append(button);
    } else if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    }
    cursor = match.index + token.length;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
}

function renderMarkdown(container, content) {
  container.textContent = "";
  let list = null;
  let code = null;
  for (const rawLine of String(content || "").split(/\r?\n/)) {
    if (rawLine.trim().startsWith("```")) {
      if (code) code = null;
      else {
        const pre = document.createElement("pre");
        code = document.createElement("code");
        pre.append(code);
        container.append(pre);
      }
      list = null;
      continue;
    }
    if (code) {
      code.append(document.createTextNode(`${rawLine}\n`));
      continue;
    }
    if (!rawLine.trim()) {
      list = null;
      continue;
    }
    const heading = rawLine.match(/^(#{1,3})\s+(.+)$/);
    const bullet = rawLine.match(/^\s*[-*]\s+(.+)$/);
    if (heading) {
      list = null;
      const node = document.createElement(`h${Math.min(heading[1].length + 1, 4)}`);
      appendInline(node, heading[2]);
      container.append(node);
    } else if (bullet) {
      if (!list) {
        list = document.createElement("ul");
        container.append(list);
      }
      const item = document.createElement("li");
      appendInline(item, bullet[1]);
      list.append(item);
    } else {
      list = null;
      const paragraph = document.createElement("p");
      appendInline(paragraph, rawLine);
      container.append(paragraph);
    }
  }
}

function setStage(index, { complete = false } = {}) {
  state.stage = Math.max(state.stage, index);
  [...$("stage-flow").children].forEach((item, position) => {
    item.classList.toggle("done", complete ? position <= index : position < index);
    item.classList.toggle("active", !complete && position === index);
  });
}

function resetStages(task) {
  state.stage = -1;
  [...$("stage-flow").children].forEach((item) => item.classList.remove("done", "active"));
  if (task.status === "succeeded") setStage(4, { complete: true });
  else if (task.status === "running") setStage(0);
}

function updateStage(type, data) {
  const detail = String(data.content || "").toLowerCase();
  if (type === "task.status") setStage(0);
  else if (type === "agent.progress" && detail.includes("research_search")) setStage(1);
  else if (type === "agent.progress" && detail.includes("research_read")) setStage(2);
  else if (type === "agent.progress" && detail.includes("research_report")) setStage(4);
  else if (type === "agent.delta") setStage(3);
  else if (type === "task.completed" && data.status === "succeeded") setStage(4, { complete: true });
}

function renderTasks() {
  const list = $("task-list");
  list.textContent = "";
  if (!state.tasks.length) {
    list.className = "task-list empty-state";
    list.textContent = "暂无任务，提交一个研究问题开始。";
    return;
  }
  list.className = "task-list";
  state.tasks.forEach((task) => {
    const button = document.createElement("button");
    button.className = `task-item ${state.selected === task.id ? "active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = taskLabel(task);
    const meta = document.createElement("div");
    const status = document.createElement("span");
    status.textContent = task.status;
    const date = document.createElement("span");
    date.textContent = shortDate(task.created_at);
    meta.append(status, date);
    button.append(title, meta);
    button.addEventListener("click", () => selectTask(task.id));
    list.append(button);
  });
}

function renderTask(task) {
  $("detail-title").textContent = taskLabel(task);
  const meta = $("task-meta");
  meta.textContent = "";
  [task.status, task.project, shortDate(task.created_at)].forEach((value, index) => {
    const pill = document.createElement("span");
    pill.className = `pill ${index === 0 ? task.status : ""}`;
    pill.textContent = value;
    meta.append(pill);
  });
  const running = task.status === "pending" || task.status === "running";
  $("cancel-task").classList.toggle("hidden", !running);
  resetStages(task);
  const answer = $("answer");
  if (task.result) {
    renderMarkdown(answer, task.result);
    answer.className = "answer markdown";
  } else if (task.error) {
    answer.textContent = task.error;
    answer.className = "answer";
  } else {
    answer.textContent = state.streamed || (running ? "Agent 正在检索证据…" : "暂无结果");
    answer.className = "answer muted";
  }
}

function addTrace(type, data) {
  updateStage(type, data);
  if (type === "agent.delta") {
    state.streamed += data.content || "";
    renderMarkdown($("answer"), state.streamed);
    $("answer").className = "answer markdown";
    return;
  }
  state.events += 1;
  $("event-count").textContent = `${state.events} events`;
  const item = document.createElement("li");
  item.className = type === "task.completed" ? "complete" : "";
  const labels = {
    "task.created": "任务已进入队列",
    "task.status": "Agent 开始执行",
    "agent.progress": "工具调用",
    "agent.stream_end": "本轮生成结束",
    "task.completed": "任务执行完成",
  };
  const detail = data.content || data.status || data.error || "";
  item.textContent = `${labels[type] || type}${detail ? ` · ${detail}` : ""}`;
  $("trace").append(item);
  $("trace").scrollTop = $("trace").scrollHeight;
}

async function loadTasks() {
  const payload = await api("/api/v1/research/tasks");
  state.tasks = payload.data;
  renderTasks();
}

async function selectTask(id) {
  state.selected = id;
  const url = new URL(window.location.href);
  url.searchParams.set("task", id);
  window.history.replaceState({}, "", url);
  state.events = 0;
  state.streamed = "";
  $("trace").textContent = "";
  $("event-count").textContent = "0 events";
  if (state.stream) state.stream.close();
  const task = (await api(`/api/v1/research/tasks/${id}`)).task;
  renderTask(task);
  renderTasks();
  if (task.status === "pending" || task.status === "running") followTask(id);
}

function followTask(id) {
  const stream = new EventSource(`/api/v1/research/tasks/${id}/events`);
  state.stream = stream;
  ["task.created", "task.status", "agent.progress", "agent.delta", "agent.stream_end", "task.completed"].forEach((type) => {
    stream.addEventListener(type, async (event) => {
      const payload = JSON.parse(event.data);
      addTrace(type, payload.data || {});
      if (type === "task.completed") {
        stream.close();
        await loadTasks();
        const task = (await api(`/api/v1/research/tasks/${id}`)).task;
        renderTask(task);
        await loadReports();
      }
    });
  });
  stream.onerror = () => {
    if (stream.readyState === EventSource.CLOSED) return;
    toast("SSE 连接中断，浏览器正在重连");
  };
}

async function loadDocuments() {
  const payload = await api("/api/v1/documents");
  const list = $("document-list");
  list.textContent = "";
  payload.data.forEach((source) => {
    const row = document.createElement("div");
    row.className = "resource";
    const title = document.createElement("strong");
    title.textContent = source.title;
    const meta = document.createElement("span");
    meta.textContent = `${source.kind.toUpperCase()} · ${source.chunks} chunks`;
    row.append(title, meta);
    list.append(row);
  });
  if (!payload.data.length) list.textContent = "尚未索引文档";
}

async function loadReports() {
  const payload = await api("/api/v1/reports");
  const list = $("report-list");
  list.textContent = "";
  payload.data.forEach((report) => {
    const row = document.createElement("div");
    row.className = "resource";
    const button = document.createElement("button");
    button.textContent = report.name;
    button.addEventListener("click", () => openReport(report.name));
    const meta = document.createElement("span");
    meta.textContent = `${Math.ceil(report.size / 1024)} KB · verified on open`;
    row.append(button, meta);
    list.append(row);
  });
  if (!payload.data.length) list.textContent = "尚未生成报告";
}

async function openReport(name) {
  const payload = await api(`/api/v1/reports/${encodeURIComponent(name)}`);
  $("report-title").textContent = `${name} · coverage ${Math.round(payload.report.verification.citation_coverage * 100)}%`;
  renderMarkdown($("report-content"), payload.report.content);
  $("report-dialog").showModal();
}

async function openSource(citation) {
  try {
    const payload = await api(`/api/v1/sources/${encodeURIComponent(citation)}`);
    const source = payload.source;
    $("source-title").textContent = source.citation;
    $("source-meta").textContent = `${source.title}${source.page ? ` · page ${source.page}` : ""}`;
    $("source-content").textContent = source.content;
    $("source-dialog").showModal();
  } catch (error) { toast(error.message); }
}

$("research-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("submit-task");
  button.disabled = true;
  try {
    const payload = await api("/api/v1/research/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: $("query").value.trim(), project: $("project").value.trim() || "default" }),
    });
    $("query").value = "";
    await loadTasks();
    await selectTask(payload.task.id);
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
});

$("file-upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  $("upload-status").textContent = `正在索引 ${file.name}…`;
  try {
    const payload = await api("/api/v1/documents", { method: "POST", body: form });
    $("upload-status").textContent = `${file.name}: ${payload.document.chunks} chunks`;
    await loadDocuments();
  } catch (error) { $("upload-status").textContent = error.message; }
  event.target.value = "";
});

$("cancel-task").addEventListener("click", async () => {
  if (!state.selected) return;
  await api(`/api/v1/research/tasks/${state.selected}/cancel`, { method: "POST" });
  toast("已请求取消任务");
});
$("refresh-tasks").addEventListener("click", loadTasks);
$("close-dialog").addEventListener("click", () => $("report-dialog").close());
$("close-source-dialog").addEventListener("click", () => $("source-dialog").close());
document.addEventListener("click", (event) => {
  const citation = event.target.closest(".citation-link")?.dataset.citation;
  if (citation) openSource(citation);
});

async function boot() {
  try {
    await api("/health");
    $("health-dot").parentElement.classList.add("online");
    $("health-text").textContent = "服务正常";
    await Promise.all([loadTasks(), loadDocuments(), loadReports()]);
    const requested = new URLSearchParams(window.location.search).get("task");
    const initial = state.tasks.find((task) => task.id === requested) || state.tasks[0];
    if (initial) await selectTask(initial.id);
  } catch (error) {
    $("health-text").textContent = "服务不可用";
    toast(error.message);
  }
}

boot();
