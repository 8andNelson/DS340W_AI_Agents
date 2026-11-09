"use strict";

// --- Helpers ---

const $ = (sel) => document.querySelector(sel);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* non-JSON */ }
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status})`);
  return body;
}

function postJson(path, data) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });
}

function fmtBytes(n) {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function fmtNum(n) { return n == null ? "—" : Number(n).toLocaleString(); }

function fmtTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function duration(startIso, endIso) {
  if (!startIso) return "";
  const s = Math.max(0, Math.round(((endIso ? new Date(endIso) : new Date()) - new Date(startIso)) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function slug(name) { return name.toLowerCase().replace(/ /g, "_"); }

const STATUS_LABELS = {
  running: "Running", starting: "Starting", success: "Finished", halted: "Halted", crashed: "Crashed",
  stopped: "Stopped", unknown: "Unknown", approved: "Approved", retrying: "Retrying", rejected: "Rejected",
  failed: "Failed",
};
const statusLabel = (s) => STATUS_LABELS[s] || (s ? s[0].toUpperCase() + s.slice(1) : "");

// Collapsible JSON tree for agent outputs.
function jsonTree(value, key = null, depth = 0) {
  const label = key === null ? "" : `<span class="k">${esc(key)}</span>: `;
  if (value === null || typeof value !== "object") {
    let cls = "s", text = JSON.stringify(value);
    if (typeof value === "number") cls = "n";
    else if (typeof value === "boolean" || value === null) cls = "b";
    else if (text.length > 400) text = JSON.stringify(value.slice(0, 400) + " …");
    return `<div class="row">${label}<span class="${cls}">${esc(text)}</span></div>`;
  }
  const isArr = Array.isArray(value);
  const entries = isArr ? value.map((v, i) => [i, v]) : Object.entries(value);
  const summary = isArr ? `[${entries.length}]` : `{${entries.length}}`;
  const inner = entries.map(([k, v]) => jsonTree(v, k, depth + 1)).join("");
  return `<details ${depth < 1 ? "open" : ""}><summary>${label}<span class="c">${summary}</span></summary>${inner}</details>`;
}

// --- Which run is shown ---
// viewRun is null for the current run, else a past run's id. It lives in the
// address bar (#papers?run=<id>) so reloads and the back button keep it.

let viewRun = null;
let currentView = "overview";

function withRun(path) {
  if (!viewRun) return path;
  return `${path}${path.includes("?") ? "&" : "?"}run=${encodeURIComponent(viewRun)}`;
}

function hashFor(view, run) {
  return `#${view}${run ? `?run=${encodeURIComponent(run)}` : ""}`;
}

function openRun(runId, view = "overview") {
  location.hash = hashFor(view, runId);
}

// --- Routing ---

const loaders = {
  overview: () => { loadDetail(); },
  agents: () => { loadDetail(); },
  papers: loadPapers,
  datasets: loadDatasets,
  scripts: loadScripts,
  documents: loadDocuments,
  activity: loadActivity,
  runs: loadRuns,
};

function route() {
  const [view, query] = (location.hash || "#overview").slice(1).split("?");
  const run = new URLSearchParams(query || "").get("run");
  const runChanged = (run || null) !== viewRun;
  viewRun = run || null;
  currentView = loaders[view] ? view : "overview";
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${currentView}`; });
  document.querySelectorAll("#nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === currentView);
    // Every page stays on the run being viewed; Run history is not per run.
    a.href = a.dataset.view === "runs" ? "#runs" : hashFor(a.dataset.view, viewRun);
  });
  $("#past-banner").hidden = !viewRun || currentView === "runs";
  $("#run-card-back").hidden = !viewRun;
  if (runChanged) resetRunScopedState();
  loaders[currentView]();
  if (runChanged) tick();
}

function resetRunScopedState() {
  status = null;
  detail = null;
  lastDetailKey = "";
  consoleOffset = 0;
  consoleRunKey = null;
  selectedAgent = null;
  selectedScriptId = null;
  selectedDoc = null;
  $("#dataset-preview").hidden = true;
  $("#console").textContent = "";
}

window.addEventListener("hashchange", route);
document.querySelectorAll("#run-card-back, #past-banner-back").forEach((el) =>
  el.addEventListener("click", (e) => { e.preventDefault(); location.hash = hashFor(currentView === "runs" ? "overview" : currentView, null); }));

// --- Status polling ---

let status = null;
let detail = null;
let lastDetailKey = "";

async function pollStatus() {
  try {
    status = await api(withRun("/api/status"));
  } catch (err) {
    $("#run-card-topic").textContent = viewRun ? err.message : "The dashboard server is not responding.";
    return;
  }
  renderRunCard();
  $("#phase-label").textContent = status.phase ? `Phase: ${status.phase}` : "";
  const ownRun = status.runner.running && !viewRun;
  $("#start-btn").disabled = status.active && !viewRun ? true : status.runner.running;
  $("#stop-btn").hidden = !ownRun;
  $("#run-note").textContent = (status.run && status.run.note) || "";
  renderPipeline();

  // Reload this run's outputs when an agent attempt has been recorded.
  const run = status.run || {};
  const key = JSON.stringify([run.run_id, run.status, status.agents.map((a) => a.live && [a.live.status, a.live.attempt])]);
  if (key !== lastDetailKey) {
    lastDetailKey = key;
    await loadDetail();
    if (["papers", "datasets", "scripts", "documents", "activity"].includes(currentView)) loaders[currentView]();
  }
}

function renderRunCard() {
  const run = status.run;
  $("#run-card-label").textContent = viewRun ? "Viewing a past run" : "Current run";
  $("#run-card").classList.toggle("past", Boolean(viewRun));
  if (!run) {
    $("#run-card-topic").textContent = "No runs yet. Enter a topic on the Overview page.";
    $("#run-card-status").hidden = true;
    $("#run-card-time").textContent = "";
    return;
  }
  $("#run-card-topic").textContent = run.topic;
  const badge = $("#run-card-status");
  badge.hidden = false;
  badge.className = `badge ${run.status}`;
  badge.textContent = statusLabel(run.status);
  $("#run-card-time").textContent = `Started ${fmtTime(run.started_at)}`;
}

async function loadDetail() {
  try { detail = await api(withRun("/api/run-detail")); } catch { detail = null; }
  renderSummary();
  if (currentView === "agents") renderAgents();
}

// --- Overview ---

function renderSummary() {
  const s = detail && detail.summary;
  if (!s || !detail.run) {
    $("#summary-row").innerHTML = "";
    return;
  }
  const tiles = [
    ["Papers found", s.papers, s.papers ? `${s.papers_verified} verified` : "Not searched yet"],
    ["Parent Paper", s.parent_paper ? "Selected" : "—", s.parent_paper || "Not selected yet"],
    ["Datasets", s.datasets, s.datasets ? `${fmtNum(s.dataset_entries)} entries`
      + (s.data_target != null ? ` (agent's target: ${fmtNum(s.data_target)})` : "") : "Not collected yet"],
    ["Master CSV rows", fmtNum(s.master_rows), s.master_rows == null ? "Not merged yet"
      : s.master_target_rows == null ? "No row target set"
      : `${s.master_target_met ? "Meets" : "Below"} the ${fmtNum(s.master_target_rows)}-row target`],
    ["Baseline", s.baseline || "—", s.evaluation ? `Evaluation: ${s.evaluation}` : "Not evaluated yet"],
    ["Experiment", s.experiment || "—", s.experiment ? "Against the baseline" : "None yet"],
  ];
  $("#summary-row").innerHTML = tiles.map(([label, value, note]) => `
    <div class="card tile">
      <div class="tile-label">${esc(label)}</div>
      <div class="tile-value">${esc(value)}</div>
      <div class="tile-note" title="${esc(note)}">${esc(note)}</div>
    </div>`).join("");
}

function agentBadge(agent) {
  if (agent.planned) return `<span class="badge">Planned</span>`;
  const live = agent.live;
  if (!live) return `<span class="badge">${status.run && status.run.active ? "Waiting" : "Not reached"}</span>`;
  return `<span class="badge ${esc(live.status)}">${esc(statusLabel(live.status))}</span>`;
}

function renderPipeline() {
  $("#pipeline").innerHTML = status.agents.map((a, i) => {
    const live = a.live || {};
    const attempt = live.attempt ? `Attempt ${live.attempt}${live.max_attempts ? ` of ${live.max_attempts}` : ""}` : "";
    const took = live.started_at ? (live.status === "running" ? `running for ${duration(live.started_at)}` : `took ${duration(live.started_at, live.finished_at)}`) : "";
    const meta = [attempt, took].filter(Boolean).join(", ");
    return `
      <div class="card agent ${a.planned ? "planned" : ""} ${live.status === "running" ? "running" : ""}" data-agent="${esc(a.name)}">
        <div class="agent-top"><span class="step">${i + 1}</span><span class="agent-name">${esc(a.name)}</span>${agentBadge(a)}</div>
        <div class="agent-role">${esc(a.role)}</div>
        ${meta ? `<div class="agent-meta">${esc(meta)}</div>` : ""}
        ${live.summary ? `<div class="agent-summary">${esc(live.summary)}</div>` : ""}
      </div>`;
  }).join("");
}

$("#pipeline").addEventListener("click", (e) => {
  const card = e.target.closest(".agent");
  if (!card || card.classList.contains("planned")) return;
  selectedAgent = card.dataset.agent;
  location.hash = hashFor("agents", viewRun);
});

$("#prompt-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#form-msg");
  msg.className = "form-msg";
  msg.textContent = "";
  $("#start-btn").disabled = true;
  try {
    await postJson("/api/run", { topic: $("#topic").value, reset: $("#reset").checked });
    msg.className = "form-msg ok";
    msg.textContent = "Run started.";
    // A new run is the current run: leave any past run and start clean.
    if (viewRun) location.hash = hashFor("overview", null);
    else { resetRunScopedState(); tick(); }
  } catch (err) {
    msg.textContent = err.message;
    $("#start-btn").disabled = false;
  }
});

$("#topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#prompt-form").requestSubmit(); }
});

$("#stop-btn").addEventListener("click", async () => {
  try { await postJson("/api/stop"); } catch (err) { $("#form-msg").textContent = err.message; }
  pollStatus();
});

// --- Console ---

let consoleOffset = 0;
let consoleRunKey = null;

function highlightConsole(text) {
  return esc(text).split("\n").map((line) => {
    if (line.startsWith("[Master Agent]")) return `<span class="hl-master">${line}</span>`;
    if (line.startsWith("DECISION: Approved")) return `<span class="hl-ok">${line}</span>`;
    if (/^DECISION: (Rejected|Retrying)|Traceback|Error|WARNING/.test(line)) return `<span class="hl-bad">${line}</span>`;
    return line;
  }).join("\n");
}

async function pollConsole() {
  const el = $("#console");
  const runKey = viewRun || (status && status.run && status.run.run_id) || "current";
  if (runKey !== consoleRunKey) {   // a different run: start its console from the top
    consoleRunKey = runKey;
    consoleOffset = 0;
    el.textContent = "";
  }
  let chunk;
  try { chunk = await api(withRun(`/api/console?offset=${consoleOffset}`)); } catch { return; }
  if (!chunk.available) {
    el.innerHTML = `<span class="muted">${status && status.run
      ? "No console was kept for this run. The console is kept for runs started from this dashboard."
      : "Start a run to see the pipeline's console here."}</span>`;
    consoleOffset = 0;
    return;
  }
  if (chunk.reset) { el.textContent = ""; }
  if (chunk.text) {
    if (!consoleOffset) el.textContent = "";
    el.insertAdjacentHTML("beforeend", highlightConsole(chunk.text));
    if ($("#autoscroll").checked) el.scrollTop = el.scrollHeight;
  } else if (!consoleOffset && !el.textContent) {
    el.innerHTML = `<span class="muted">Waiting for output…</span>`;
  }
  consoleOffset = chunk.offset;
}

// --- Agent outputs ---

let selectedAgent = null;
let selectedAttempt = null;

function renderAgents() {
  if (!status) return;
  const agents = status.agents.filter((a) => !a.planned);
  if (!selectedAgent || !agents.some((a) => a.name === selectedAgent)) selectedAgent = agents[0].name;
  $("#agent-list").innerHTML = agents.map((a) => `
    <button class="list-item ${a.name === selectedAgent ? "active" : ""}" data-agent="${esc(a.name)}">
      <span class="list-item-row"><strong>${esc(a.name)}</strong>${agentBadge(a)}</span>
      <span class="sub">${esc(a.phase)}</span>
    </button>`).join("");
  renderAgentDetail(agents.find((a) => a.name === selectedAgent));
}

$("#agent-list").addEventListener("click", (e) => {
  const item = e.target.closest(".list-item");
  if (!item) return;
  selectedAgent = item.dataset.agent;
  selectedAttempt = null;
  renderAgents();
});

function renderAgentDetail(agent) {
  const attempts = ((detail && detail.attempts) || []).filter((a) => a.agent === agent.name);
  if (selectedAttempt == null || selectedAttempt >= attempts.length) selectedAttempt = attempts.length - 1;
  const chosen = attempts[selectedAttempt];
  const scriptsLink = ["Replication Agent", "Experiment Agent"].includes(agent.name)
    ? `<a class="btn sm" href="${hashFor("scripts", viewRun)}">View the scripts it wrote</a>` : "";

  $("#agent-detail").innerHTML = `
    <div class="card">
      <div class="agent-detail-head">
        <div>
          <h3>${esc(agent.name)}</h3>
          <div class="muted">${esc(agent.role)}</div>
        </div>
        ${scriptsLink}
      </div>
    </div>
    ${!attempts.length ? `<div class="empty">${status.run && status.run.active
        ? "This agent has not finished an attempt in this run yet."
        : "This agent did not run in this run."}</div>` : `
    <div class="tabs" id="attempt-tabs">${attempts.map((a, i) => `
      <button class="chip ${i === selectedAttempt ? "active" : ""}" data-attempt="${i}">
        Attempt ${a.attempt}: ${esc(a.decision)}
      </button>`).join("")}
    </div>
    <div class="card">
      <h3>Report</h3>
      <div class="muted small">Recorded ${esc(fmtTime(chosen.recorded_at))}. Master Agent decision: ${esc(chosen.decision)}.</div>
      <pre class="report">${esc(chosen.report_text || "No report text was recorded for this attempt.")}</pre>
    </div>
    <div class="card">
      <h3>Structured output</h3>
      ${chosen.output == null ? `<div class="muted">No output was recorded for this attempt (the agent raised an error).</div>`
        : `<div class="json">${jsonTree(chosen.output)}</div>`}
    </div>`}`;
}

$("#agent-detail").addEventListener("click", (e) => {
  const tab = e.target.closest("[data-attempt]");
  if (!tab) return;
  selectedAttempt = Number(tab.dataset.attempt);
  renderAgents();
});

// --- Scripts ---

const PY_KEYWORDS = new Set(("False None True and as assert async await break class continue def del elif else " +
  "except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
const PY_TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfuRBFU]{0,2}"(?:\\.|[^"\\\n])*"|[rbfuRBFU]{0,2}'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b|([A-Za-z_][A-Za-z0-9_]*)/g;

// Highlighted HTML for each line. A token that spans lines (a docstring) is
// split so every line stays a self-contained row.
function highlightLines(text, language) {
  const lines = [""];
  const push = (chunk, cls) => {
    chunk.split("\n").forEach((part, i) => {
      if (i > 0) lines.push("");
      if (part) lines[lines.length - 1] += cls ? `<span class="${cls}">${esc(part)}</span>` : esc(part);
    });
  };
  if (language !== "python") { push(text, ""); return lines; }
  let last = 0, m, prevWord = "";
  PY_TOKEN.lastIndex = 0;
  while ((m = PY_TOKEN.exec(text))) {
    push(text.slice(last, m.index), "");
    let cls = "";
    if (m[1]) cls = "tk-com";
    else if (m[2]) cls = "tk-str";
    else if (m[3]) cls = "tk-num";
    else if (PY_KEYWORDS.has(m[4])) cls = "tk-kw";
    else if (prevWord === "def" || prevWord === "class") cls = "tk-def";
    push(m[0], cls);
    if (m[4]) prevWord = m[4];
    last = m.index + m[0].length;
  }
  push(text.slice(last), "");
  return lines;
}

// Line diff (longest common subsequence). Returns [{op: " " | "+" | "-", a, b, html}].
function diffLines(baseText, newText, language) {
  const a = baseText.split("\n"), b = newText.split("\n");
  if (a.length * b.length > 4_000_000) return null;
  const ha = highlightLines(baseText, language), hb = highlightLines(newText, language);
  const n = a.length, m = b.length;
  const lcs = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const rows = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { rows.push({ op: " ", a: i + 1, b: j + 1, html: hb[j] }); i++; j++; }
    else if (i < n && (j >= m || lcs[i + 1][j] >= lcs[i][j + 1])) { rows.push({ op: "-", a: i + 1, b: "", html: ha[i] }); i++; }
    else { rows.push({ op: "+", a: "", b: j + 1, html: hb[j] }); j++; }
  }
  return rows;
}

let scriptData = null;
let selectedScriptId = null;
let scriptMode = "code";

async function loadScripts() {
  const list = $("#script-list");
  try {
    scriptData = await api(withRun("/api/scripts"));
  } catch (err) {
    list.innerHTML = `<div class="muted small">${esc(err.message)}</div>`;
    return;
  }
  const files = scriptData.groups.flatMap((g) => g.files);
  if (!files.length) {
    list.innerHTML = "";
    $("#script-detail").innerHTML = `<div class="empty">This run has no scripts yet. Scripts appear once its Replication Agent has run.</div>`;
    return;
  }
  if (!files.some((f) => f.id === selectedScriptId)) selectedScriptId = files[0].id;
  list.innerHTML = scriptData.groups.map((g) => `
    <div class="list-group">
      <div class="list-group-head">${esc(g.agent)}</div>
      ${g.files.map((f) => `
        <button class="list-item ${f.id === selectedScriptId ? "active" : ""} ${f.is_code ? "" : "secondary"}" data-script="${f.id}">
          <span class="list-item-row"><strong class="${f.is_code ? "mono" : ""}">${esc(f.name)}</strong></span>
          <span class="sub">${esc(f.role)}</span>
        </button>`).join("")}
    </div>`).join("");
  showScript(selectedScriptId);
}

$("#script-list").addEventListener("click", (e) => {
  const item = e.target.closest("[data-script]");
  if (!item) return;
  selectedScriptId = Number(item.dataset.script);
  document.querySelectorAll("#script-list .list-item").forEach((el) => el.classList.toggle("active", el === item));
  scriptMode = "code";
  showScript(selectedScriptId);
});

function findScript(id) {
  for (const g of scriptData.groups) {
    const f = g.files.find((x) => x.id === id);
    if (f) return { file: f, group: g };
  }
  return {};
}

function provenance(group) {
  const c = group.context || {};
  const models = Array.isArray(c.generated_by) ? c.generated_by.join(", ") : c.generated_by;
  const fields = [
    ["Written by", models ? `${models} (the model that answered)` : (c.source === "cloned" ? "The paper's authors (cloned repository)" : "")],
    ["Source", c.repo_url ? `${c.repo_url}${c.commit_sha ? ` at commit ${String(c.commit_sha).slice(0, 10)}` : ""}` : (c.source || "")],
    ["Entry point", c.entry_point],
    ["Experiment", c.category ? c.category.replace(/_/g, " ") : ""],
    ["Hypothesis", c.hypothesis],
    ["What changed", c.what_changed],
    ["Result against baseline", c.overall],
    ["Exit code", c.exit_code],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");
  const assumptions = c.inferred_assumptions || [];
  return `
    <dl class="prov">${fields.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
    ${assumptions.length ? `<details class="assumptions"><summary>${assumptions.length} inferred assumptions (reconstructed code, not the authors')</summary>
      <ul>${assumptions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul></details>` : ""}`;
}

const DISK_NOTES = {
  changed: "A later run has changed this file on disk. You are seeing the version this run recorded.",
  missing: "This file is no longer on disk. You are seeing the version this run recorded.",
};

async function showScript(id) {
  const { file, group } = findScript(id);
  if (!file) return;
  $("#script-detail").innerHTML = `
    <div class="card script-card">
      <div class="script-bar">
        <div class="script-title">
          <span class="mono">${esc(file.path)}</span>
          <span class="muted small">${esc(file.role)}, ${esc(fmtBytes(file.size_bytes))}</span>
        </div>
        <div class="btn-group">
          <button class="btn sm" id="script-copy">Copy</button>
          <a class="btn sm" href="/api/artifacts/${file.id}/download">Download</a>
          <button class="btn sm primary" id="script-open">Open in VS Code</button>
        </div>
      </div>
      <div class="note" id="script-note" role="status">${esc(DISK_NOTES[file.disk] || "")}</div>
      ${provenance(group)}
      ${file.baseline_id ? `
        <div class="tabs" id="script-tabs">
          <button class="chip ${scriptMode === "code" ? "active" : ""}" data-mode="code">Code</button>
          <button class="chip ${scriptMode === "diff" ? "active" : ""}" data-mode="diff">Changes from baseline</button>
        </div>` : ""}
      <div class="code-view" id="code-view"><div class="muted small" style="padding:12px">Loading…</div></div>
    </div>`;

  let script, baseline = null;
  try {
    script = await api(`/api/scripts/${file.id}`);
    if (scriptMode === "diff" && file.baseline_id) baseline = await api(`/api/scripts/${file.baseline_id}`);
  } catch (err) {
    $("#code-view").innerHTML = `<div class="note err" style="padding:12px">${esc(err.message)}</div>`;
    return;
  }
  if (selectedScriptId !== id) return;  // another script was picked meanwhile
  renderCode(script, baseline);

  $("#script-copy").onclick = async () => {
    try { await navigator.clipboard.writeText(script.text); $("#script-note").textContent = "Copied to the clipboard."; }
    catch { $("#script-note").textContent = "The browser did not allow copying. Use Download instead."; }
  };
  $("#script-open").onclick = async () => {
    const note = $("#script-note");
    note.className = "note";
    try {
      const r = await postJson(`/api/scripts/${file.id}/open`);
      note.textContent = r.is_copy
        ? `Opened a copy of the recorded version in VS Code: ${r.opened}. Edits there do not change the pipeline's file.`
        : `Opened ${r.opened} in VS Code.`;
    } catch (err) {
      note.className = "note err";
      note.textContent = err.message;
    }
  };
}

function renderCode(script, baseline) {
  const view = $("#code-view");
  const truncated = script.truncated ? `<div class="note" style="padding:8px 12px">Showing the first 2 MB. Download the file to see all of it.</div>` : "";
  if (baseline) {
    const rows = diffLines(baseline.text, script.text, script.language);
    if (!rows) { view.innerHTML = `<div class="note" style="padding:12px">These files are too large to compare here.</div>`; return; }
    const changed = rows.filter((r) => r.op !== " ").length;
    const CONTEXT = 3;
    const keep = rows.map((r, i) => rows.slice(Math.max(0, i - CONTEXT), i + CONTEXT + 1).some((x) => x.op !== " "));
    let html = "", hidden = 0;
    rows.forEach((r, i) => {
      if (!keep[i]) { hidden++; return; }
      if (hidden) { html += `<tr class="gap"><td colspan="4">${hidden} unchanged lines</td></tr>`; hidden = 0; }
      const cls = r.op === "+" ? "add" : r.op === "-" ? "del" : "";
      html += `<tr class="${cls}"><td class="ln">${r.a}</td><td class="ln">${r.b}</td><td class="op">${r.op.trim()}</td><td class="src">${r.html || " "}</td></tr>`;
    });
    if (hidden) html += `<tr class="gap"><td colspan="4">${hidden} unchanged lines</td></tr>`;
    view.innerHTML = `<div class="diff-summary">${changed ? `${changed} lines differ from the baseline script <span class="mono">${esc(baseline.name)}</span>.` : "This script is the same as the baseline."}</div>
      <table class="code diff"><tbody>${html}</tbody></table>`;
  } else {
    const lines = highlightLines(script.text, script.language);
    view.innerHTML = truncated + `<table class="code"><tbody>${lines.map((l, i) =>
      `<tr><td class="ln">${i + 1}</td><td class="src">${l || " "}</td></tr>`).join("")}</tbody></table>`;
  }
}

$("#script-detail").addEventListener("click", (e) => {
  const tab = e.target.closest("[data-mode]");
  if (!tab) return;
  scriptMode = tab.dataset.mode;
  showScript(selectedScriptId);
});

// --- Papers ---

async function loadPapers() {
  const el = $("#papers");
  let papers;
  try { papers = await api(withRun("/api/papers")); } catch (err) { el.innerHTML = `<div class="empty">${esc(err.message)}</div>`; return; }
  if (!papers.length) {
    el.innerHTML = `<div class="empty">${status && status.run
      ? "This run has no papers yet. They appear once its Research Agent finishes."
      : "No runs yet. Start one from the Overview page."}</div>`;
    return;
  }
  papers.sort((a, b) => (b.is_parent - a.is_parent));
  el.innerHTML = papers.map((p) => {
    const vs = (p.validation_status || "").toLowerCase();
    const authors = (p.authors || []).slice(0, 4).join(", ") + ((p.authors || []).length > 4 ? " et al." : "");
    const meta = [authors || "Authors unknown", p.year, p.venue || "Unknown venue", p.doi && `DOI ${p.doi}`].filter(Boolean);
    return `
      <article class="card paper ${p.is_parent ? "is-parent" : ""}">
        <div class="paper-head">
          <div>
            <div class="paper-title">${esc(p.title)}</div>
            <div class="paper-meta">${meta.map(esc).join(", ")}</div>
          </div>
          <div class="paper-badges">
            ${p.is_parent ? `<span class="badge parent">Parent Paper</span>` : ""}
            ${vs ? `<span class="badge ${esc(vs)}">${esc(p.validation_status)}</span>` : ""}
            ${p.has_code ? `<span class="badge">Has code</span>` : ""}
          </div>
        </div>
        <dl class="paper-body">
          <div><dt>Dataset</dt><dd>${esc(p.dataset || "Not stated")}</dd></div>
          <div><dt>Methodology</dt><dd>${esc(p.methodology || "Not stated")}</dd></div>
          <div><dt>Results</dt><dd>${esc(p.results_summary || "Not stated")}</dd></div>
          ${p.rejection_reason ? `<div><dt>Rejection reason</dt><dd>${esc(p.rejection_reason)}</dd></div>` : ""}
        </dl>
        ${p.abstract ? `<details><summary>Abstract</summary><p>${esc(p.abstract)}</p></details>` : ""}
        <div class="paper-actions">
          ${p.paper_url ? `<a class="btn sm" href="${esc(p.paper_url)}" target="_blank" rel="noopener noreferrer">Publisher page</a>` : ""}
          ${p.code_url ? `<a class="btn sm ghost" href="${esc(p.code_url)}" target="_blank" rel="noopener noreferrer">Code</a>` : ""}
          ${p.pdf_name
            ? `<button class="btn sm primary" data-view-pdf="/api/pdfs/${esc(p.pdf_name)}/view">View PDF</button>
               <a class="btn sm ghost" href="/api/pdfs/${esc(p.pdf_name)}/download">Download PDF</a>`
            : `<button class="btn sm" data-fetch-pdf="${p.index}">Get open-access PDF</button>`}
          <span class="note" data-note="${p.index}"></span>
        </div>
      </article>`;
  }).join("");
}

$("#papers").addEventListener("click", async (e) => {
  const view = e.target.closest("[data-view-pdf]");
  if (view) { selectedDoc = view.dataset.viewPdf; location.hash = hashFor("documents", viewRun); return; }
  const fetchBtn = e.target.closest("[data-fetch-pdf]");
  if (!fetchBtn) return;
  const idx = fetchBtn.dataset.fetchPdf;
  const note = document.querySelector(`[data-note="${idx}"]`);
  fetchBtn.disabled = true;
  note.className = "note";
  note.textContent = "Looking for an open-access PDF…";
  try {
    await postJson(withRun(`/api/papers/${idx}/pdf`));
    loadPapers();
  } catch (err) {
    note.className = "note err";
    note.textContent = err.message;
    fetchBtn.disabled = false;
  }
});

// --- Datasets ---

const preview = { id: null, name: "", offset: 0, limit: 100 };

async function loadDatasets() {
  const table = $("#dataset-table");
  let files;
  try { files = await api(withRun("/api/datasets")); } catch (err) { table.innerHTML = `<tr><td>${esc(err.message)}</td></tr>`; return; }
  if (!files.length) {
    table.innerHTML = `<tr><td class="muted">${status && status.run
      ? "This run has no datasets yet. They appear once its Data Agent finishes."
      : "No runs yet."}</td></tr>`;
    return;
  }
  const groups = { processed: "Built by the Cleaning Agent (master dataset and splits)", raw: "Downloaded by the Data Agent" };
  let rows = "";
  for (const group of ["processed", "raw"]) {
    const items = files.filter((f) => f.group === group);
    if (!items.length) continue;
    rows += `<tr class="group"><td colspan="5">${esc(groups[group])}</td></tr>`;
    rows += items.map((f) => `
      <tr>
        <td><strong>${esc(f.display_name || f.name)}</strong><div class="muted small">${esc(f.path)}</div>
          ${f.disk !== "same" ? `<div class="note small">${f.disk === "missing" ? "No longer on disk; served from this run's record." : "Changed on disk by a later run; served from this run's record."}</div>` : ""}</td>
        <td>${f.source_url ? `<a href="${esc(f.source_url)}" target="_blank" rel="noopener noreferrer">${esc(f.source || "Source")}</a>` : esc(f.source || "")}</td>
        <td class="num">${fmtNum(f.entry_count)}</td>
        <td class="num">${esc(fmtBytes(f.size_bytes))}</td>
        <td><div class="btn-group">
          <button class="btn sm" data-preview="${f.id}" data-name="${esc(f.display_name || f.name)}">Preview</button>
          <a class="btn sm ghost" href="/api/artifacts/${f.id}/download">Download</a>
        </div></td>
      </tr>`).join("");
  }
  table.innerHTML = `<thead><tr><th>Dataset</th><th>Source</th><th class="num">Rows</th><th class="num">Size</th><th></th></tr></thead><tbody>${rows}</tbody>`;
}

$("#dataset-table").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-preview]");
  if (!btn) return;
  Object.assign(preview, { id: btn.dataset.preview, name: btn.dataset.name, offset: 0 });
  loadPreview();
});

async function loadPreview() {
  const box = $("#dataset-preview");
  box.hidden = false;
  $("#preview-title").textContent = preview.name;
  $("#preview-download").href = `/api/artifacts/${preview.id}/download`;
  let data;
  try {
    data = await api(`/api/artifacts/${preview.id}/preview?offset=${preview.offset}&limit=${preview.limit}`);
  } catch (err) {
    $("#preview-table").innerHTML = `<tr><td>${esc(err.message)}</td></tr>`;
    return;
  }
  const first = data.offset + 1, last = data.offset + data.rows.length;
  $("#preview-meta").textContent = data.rows.length
    ? `Rows ${fmtNum(first)} to ${fmtNum(last)}, ${data.columns.length} columns${data.source === "database" ? ", from this run's record" : ""}`
    : "No rows in this range.";
  $("#preview-prev").disabled = data.offset === 0;
  $("#preview-next").disabled = !data.has_more;
  $("#preview-table").innerHTML =
    `<thead><tr><th>#</th>${data.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
     <tbody>${data.rows.map((r, i) => `<tr><td class="idx">${data.offset + i + 1}</td>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#preview-prev").addEventListener("click", () => { preview.offset = Math.max(0, preview.offset - preview.limit); loadPreview(); });
$("#preview-next").addEventListener("click", () => { preview.offset += preview.limit; loadPreview(); });
$("#preview-close").addEventListener("click", () => { $("#dataset-preview").hidden = true; });

// --- Documents ---

let selectedDoc = null;
let documents = [];

async function loadDocuments() {
  try { documents = await api(withRun("/api/documents")); } catch (err) { $("#pdf-list").innerHTML = `<div class="muted">${esc(err.message)}</div>`; return; }
  if (!documents.length) {
    $("#pdf-list").innerHTML = "";
    $("#pdf-detail").innerHTML = `<div class="empty">This run has no documents yet. Fetch a paper's open-access PDF from the Papers page; the report and slides will appear here once those agents exist.</div>`;
    return;
  }
  if (!documents.some((d) => d.view_url === selectedDoc)) selectedDoc = documents[0].view_url;
  $("#pdf-list").innerHTML = documents.map((d) => `
    <button class="list-item ${d.view_url === selectedDoc ? "active" : ""}" data-doc="${esc(d.view_url)}">
      <strong>${esc(d.name)}</strong>
      <span class="sub">${esc(d.origin)}, ${esc(fmtBytes(d.size_bytes))}</span>
    </button>`).join("");
  showDocument(selectedDoc);
}

function showDocument(viewUrl) {
  const doc = documents.find((d) => d.view_url === viewUrl);
  if (!doc) return;
  selectedDoc = viewUrl;
  document.querySelectorAll("#pdf-list .list-item").forEach((el) => el.classList.toggle("active", el.dataset.doc === viewUrl));
  $("#pdf-detail").innerHTML = `
    <div class="pdf-bar">
      <span class="name">${esc(doc.name)}</span>
      <div class="spacer"></div>
      <a class="btn sm" href="${esc(doc.view_url)}" target="_blank" rel="noopener">Open in new tab</a>
      <a class="btn sm primary" href="${esc(doc.download_url)}">Download</a>
    </div>
    <iframe src="${esc(doc.view_url)}" title="${esc(doc.name)}"></iframe>`;
}

$("#pdf-list").addEventListener("click", (e) => {
  const item = e.target.closest("[data-doc]");
  if (item) showDocument(item.dataset.doc);
});

// --- Master Agent log ---

let activityFilter = "all";

async function loadActivity() {
  const el = $("#activity");
  let entries;
  try { entries = await api(withRun("/api/activity")); } catch (err) { el.innerHTML = `<div class="empty">${esc(err.message)}</div>`; return; }
  const shown = entries.filter((e) => activityFilter === "all" || e.action === activityFilter);
  if (!shown.length) { el.innerHTML = `<div class="empty">The Master Agent has made no decisions in this run yet.</div>`; return; }
  el.innerHTML = shown.map((e) => `
    <div class="card event">
      <div class="event-time">${esc(fmtTime(e.timestamp))}</div>
      <div>
        <div class="event-head">
          <span class="badge ${e.decision === "Approved" ? "approved" : e.decision === "Retrying" ? "retrying" : "rejected"}">${esc(e.decision)}</span>
          <strong>${esc(e.action)}</strong><span class="muted">for the ${esc(e.target)}</span>
        </div>
        <div class="event-issue">${esc(e.issue)}</div>
        <div class="event-next">Next: ${esc(e.next)}</div>
      </div>
    </div>`).join("");
}

$("#activity-filters").addEventListener("click", (e) => {
  const chip = e.target.closest("[data-filter]");
  if (!chip) return;
  activityFilter = chip.dataset.filter;
  document.querySelectorAll("#activity-filters .chip").forEach((c) => c.classList.toggle("active", c === chip));
  loadActivity();
});

// --- Run history ---

async function loadRuns() {
  const el = $("#run-list");
  let runs;
  try { runs = await api("/api/runs"); } catch (err) { el.innerHTML = `<div class="empty">${esc(err.message)}</div>`; return; }
  if (!runs.length) { el.innerHTML = `<div class="empty">No runs yet. Start one from the Overview page.</div>`; return; }
  el.innerHTML = runs.map((r) => {
    const isViewed = r.is_current ? !viewRun : r.run_id === viewRun;
    const reached = r.final_phase ? `Reached ${r.final_phase.replace(/_/g, " ").toLowerCase()}` : "";
    return `
      <article class="card run-row ${isViewed ? "viewed" : ""}">
        <div class="run-row-main">
          <div class="run-row-head">
            <span class="badge ${esc(r.status)}">${esc(statusLabel(r.status))}</span>
            ${r.is_current ? `<span class="badge current">Current run</span>` : ""}
            <span class="muted small">${esc(fmtTime(r.started_at))}${r.finished_at ? `, took ${esc(duration(r.started_at, r.finished_at))}` : ""}</span>
          </div>
          <div class="run-row-topic">${esc(r.topic)}</div>
          <div class="muted small">${esc([r.parent_paper_title ? `Parent Paper: ${r.parent_paper_title}` : "No Parent Paper selected", reached].filter(Boolean).join(". "))}</div>
        </div>
        <div class="run-row-actions">
          ${isViewed ? `<span class="muted small">Showing now</span>` : `<button class="btn sm primary" data-open-run="${r.is_current ? "" : esc(r.run_id)}">Open this run</button>`}
          <div class="run-row-links">
            ${["papers", "datasets", "scripts", "documents"].map((v) =>
              `<a href="${hashFor(v, r.is_current ? null : r.run_id)}">${v[0].toUpperCase() + v.slice(1)}</a>`).join("")}
          </div>
        </div>
      </article>`;
  }).join("");
}

$("#run-list").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-open-run]");
  if (btn) openRun(btn.dataset.openRun || null);
});

// --- Model router (FreeLLMAPI in Docker) ---

const ROUTER_LABELS = {
  running: "Running. The agents can reach their models.",
  starting: "Starting",
  stopping: "Stopping",
  stopped: "Stopped. Start it before you start a run.",
  idle: "Not running. Start it before you start a run.",
  error: "Could not start",
};

async function pollRouter() {
  let r;
  try { r = await api("/api/freellmapi"); } catch { return; }
  const box = $("#router");
  box.hidden = !r.enabled;
  if (!r.enabled) return;
  const state = r.busy ? r.state : (r.healthy ? "running" : r.state === "error" ? "error" : (r.state === "idle" ? "idle" : "stopped"));
  $("#router-dot").className = `router-dot ${state}`;
  const label = ROUTER_LABELS[state] || state;
  $("#router-msg").textContent = r.busy || state === "error" ? `${label}: ${r.message}` : label;
  $("#router-open").href = r.url;
  $("#router-open").hidden = !r.healthy;
  $("#router-start").hidden = r.healthy || r.busy;
  $("#router-stop").hidden = !r.healthy || r.busy;
  $("#router-log").textContent = r.log.join("\n");
}

async function routerAction(action) {
  try { await postJson(`/api/freellmapi/${action}`); }
  catch (err) { $("#router-msg").textContent = err.message; return; }
  pollRouter();
}

$("#router-start").addEventListener("click", () => routerAction("start"));
$("#router-stop").addEventListener("click", () => routerAction("stop"));

// --- Start ---

async function tick() {
  await pollStatus();
  if (currentView === "overview") await Promise.all([pollConsole(), pollRouter()]);
}

route();
tick();
setInterval(tick, 2000);
