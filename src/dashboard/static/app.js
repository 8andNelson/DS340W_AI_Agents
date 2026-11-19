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
  return elapsedSeconds(((endIso ? new Date(endIso) : new Date()) - new Date(startIso)) / 1000);
}

function elapsedSeconds(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function slug(name) { return name.toLowerCase().replace(/ /g, "_"); }

const STATUS_LABELS = {
  running: "Running", starting: "Starting", success: "Finished", halted: "Halted", crashed: "Crashed",
  stopped: "Stopped", unknown: "Unknown", approved: "Approved", retrying: "Retrying", rejected: "Rejected",
  failed: "Failed", checking: "Checking", interrupted: "Interrupted", skipped: "Turned off",
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
  report: loadReport,
  activity: loadActivity,
  runs: loadRuns,
  compare: loadCompare,
  memory: loadMemory,
};
// Views that cover every run, not the run being viewed.
const GLOBAL_VIEWS = ["runs", "compare", "memory"];

$("#nav-toggle").addEventListener("click", () => {
  const open = !document.body.classList.contains("nav-open");
  document.body.classList.toggle("nav-open", open);
  $("#nav-toggle").setAttribute("aria-expanded", String(open));
});

function route() {
  document.body.classList.remove("nav-open");
  $("#nav-toggle").setAttribute("aria-expanded", "false");
  const [view, query] = (location.hash || "#overview").slice(1).split("?");
  const run = new URLSearchParams(query || "").get("run");
  const runChanged = (run || null) !== viewRun;
  viewRun = run || null;
  currentView = loaders[view] ? view : "overview";
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${currentView}`; });
  document.querySelectorAll("#nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === currentView);
    // Every page stays on the run being viewed; Run history, Compare runs and Memory are not per run.
    a.href = GLOBAL_VIEWS.includes(a.dataset.view) ? `#${a.dataset.view}` : hashFor(a.dataset.view, viewRun);
  });
  $("#past-banner").hidden = !viewRun || GLOBAL_VIEWS.includes(currentView);
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
  reportData = null;
  deck = null;
  slideIndex = 0;
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
  if (status) renderPipeline();
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
    <div class="tile">
      <div class="tile-label">${esc(label)}</div>
      <div class="tile-value">${esc(value)}</div>
      <div class="tile-note" title="${esc(note)}">${esc(note)}</div>
    </div>`).join("");
}

function agentBadge(agent) {
  if (agent.planned) return `<span class="badge">Planned</span>`;
  const live = agent.live;
  if (!live) return `<span class="badge">${status.run && status.run.active ? "Waiting" : "Not reached"}</span>`;
  const inferred = live.timing_method === "inferred" ? " (inferred)" : "";
  return `<span class="badge ${esc(live.status)}">${esc(statusLabel(live.status) + inferred)}</span>`;
}

// The first stages a supplied Parent Paper replaces (the server's PIPELINE names).
const PAPER_SEARCH_STAGES = ["Intake Agent", "Research Agent", "Validation Agent"];
const PARENT_INTAKE = { name: "Parent Paper Intake", phase: "MASTER_APPROVAL",
  role: "Uses your PDF and prediction target; the paper search is skipped." };

function chosenPdf() { return $("#parent-pdf").files[0] || null; }

// With a PDF chosen and no run in progress, the pipeline shows the next run's stages.
function previewingNextRun() { return Boolean(chosenPdf()) && !viewRun && !(status && status.active); }

function pipelineAgents() {
  if (!previewingNextRun()) return status.agents;
  return [{ ...PARENT_INTAKE, live: null, preview: true },
    ...status.agents.filter((a) => !PAPER_SEARCH_STAGES.includes(a.name) && a.name !== PARENT_INTAKE.name)
      .map((a) => ({ ...a, live: null, preview: true }))];
}

function agentSwitch(name) {
  const s = switches[name];
  if (!s) return "";
  return `
    <button type="button" class="switch card-switch" role="switch" aria-checked="${s.on}" data-switch="${esc(name)}"
      ${s.available ? "" : "disabled"} title="${esc(s.note || "Whether this agent runs in the next run you start")}">
      <span class="switch-track" aria-hidden="true"><span class="switch-knob"></span></span>
      <span class="switch-state">${s.on ? "On" : "Off"} next run</span>
    </button>`;
}

// The agent's last attempt in plain words (src/dashboard/plain.py), when its outputs are loaded.
function stageHeadline(name) {
  const mine = ((detail && detail.attempts) || []).filter((t) => t.agent === name && t.plain);
  return mine.length ? mine[mine.length - 1].plain.headline : "";
}

function stageState(a) {
  if (a.preview) return "preview";
  if (a.planned) return "planned";
  return (a.live && a.live.status) || "waiting";
}

function renderPipeline() {
  const preview = previewingNextRun();
  $("#pipeline-note").textContent = preview ? "Preview of the next run with your PDF." : "";
  $("#pipeline").innerHTML = `<ol class="track">${pipelineAgents().map((a, i) => {
    const live = a.live || {};
    const attempt = live.attempt ? `Attempt ${live.attempt}${live.max_attempts ? ` of ${live.max_attempts}` : ""}` : "";
    let took = "";
    if (live.status === "running" && live.started_at) took = `running for ${duration(live.started_at)}`;
    else if (live.duration_seconds != null) took = `execution took ${elapsedSeconds(live.duration_seconds)}`;
    else if (live.timing_method === "legacy_estimate" && live.started_at && live.finished_at)
      took = `estimated ${duration(live.started_at, live.finished_at)}`;
    else if (live.status === "interrupted") took = "execution duration unknown";
    const meta = [attempt, took].filter(Boolean).join(", ");
    const busy = ["running", "checking"].includes(live.status);
    const headline = preview || busy ? "" : stageHeadline(a.name);
    const state = stageState(a);
    const opens = !a.planned && !a.preview;
    return `
      <li class="stage ${esc(state)}" data-agent="${esc(a.name)}">
        <span class="node" aria-hidden="true"></span>
        <div class="stage-body">
          <div class="stage-top">
            ${opens ? `<a class="stage-name" href="${hashFor("agents", viewRun)}" data-open="${esc(a.name)}"><span class="stage-num">${i + 1}</span>${esc(a.name)}</a>`
              : `<span class="stage-name"><span class="stage-num">${i + 1}</span>${esc(a.name)}</span>`}
            ${a.preview ? `<span class="badge">Next run</span>` : agentBadge(a)}
          </div>
          <p class="stage-text">${esc(headline || a.role)}</p>
          ${meta ? `<p class="stage-meta">${esc(meta)}</p>` : ""}
        </div>
        ${agentSwitch(a.name)}
      </li>`;
  }).join("")}</ol>`;
}

$("#pipeline").addEventListener("click", async (e) => {
  const button = e.target.closest("button[data-switch]");
  if (button) {  // an optional agent's switch, not a link to its outputs
    if (button.disabled) return;
    button.disabled = true;
    try {
      const on = button.getAttribute("aria-checked") !== "true";
      setSwitches((await postJson("/api/agent-switches", { name: button.dataset.switch, on })).agents);
    } catch (err) {
      $("#form-msg").textContent = err.message;
    }
    renderPipeline();
    return;
  }
  const card = e.target.closest(".stage");
  if (!card || card.classList.contains("planned") || card.classList.contains("preview")) return;
  e.preventDefault();
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
    const body = { topic: $("#topic").value, reset: $("#reset").checked };
    const file = chosenPdf();
    if (file) {
      const data = new FormData();
      data.append("pdf", file);
      data.append("topic", body.topic);
      data.append("title", $("#parent-title").value);
      data.append("target_terms", $("#parent-target").value);
      msg.textContent = "Checking Parent Paper PDF…";
      const response = await fetch("/api/parent-paper", { method: "POST", body: data });
      const uploaded = await response.json();
      if (!response.ok) throw new Error(uploaded.error || "The PDF could not be uploaded.");
      body.parent_paper_id = uploaded.parent_paper_id;
      $("#parent-input-status").textContent = `Fixed input ${uploaded.input_fingerprint.slice(0, 12)} · ${uploaded.pages} page(s)`;
    }
    await postJson("/api/run", body);
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

// "Use my Parent Paper PDF" opens the file explorer; choosing a file sets up the run.
$("#pick-pdf").addEventListener("click", () => $("#parent-pdf").click());

function showChosenPdf() {
  const file = chosenPdf();
  $("#parent-pdf-fields").hidden = !file;
  $("#pick-pdf").textContent = file ? "Choose another PDF…" : "Use my Parent Paper PDF…";
  $("#pick-pdf-hint").hidden = Boolean(file);
  for (const id of ["parent-title", "parent-target"]) $("#" + id).required = Boolean(file);
  $("#pdf-name").textContent = file ? `${file.name} (${(file.size / 1048576).toFixed(1)} MB)` : "";
  $("#parent-input-status").textContent = "";
  if (file && !$("#parent-title").value) $("#parent-title").focus();
  if (status) renderPipeline();
}

$("#parent-pdf").addEventListener("change", showChosenPdf);
$("#pdf-clear").addEventListener("click", () => { $("#parent-pdf").value = ""; showChosenPdf(); });

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
// Simple view: the plain summary first, the technical report and JSON folded.
// Remembered in this browser only (a per-viewer preference).
let simpleView = true;
try { simpleView = localStorage.getItem("simpleView") !== "off"; } catch (err) { /* storage blocked */ }
$("#simple-view").checked = simpleView;
$("#simple-view").addEventListener("change", (e) => {
  simpleView = e.target.checked;
  try { localStorage.setItem("simpleView", simpleView ? "on" : "off"); } catch (err) { /* storage blocked */ }
  renderAgents();
});

function plainCard(plain) {
  if (!plain) return "";
  return `
    <div class="card plain">
      <h3>In plain words</h3>
      <p class="plain-headline">${esc(plain.headline || "")}</p>
      ${(plain.points || []).length ? `<ul class="plain-points">${plain.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
      ${plain.decision ? `<p class="plain-decision">${esc(plain.decision)}</p>` : ""}
      ${plain.next ? `<p class="plain-next"><strong>Next:</strong> ${esc(plain.next)}</p>` : ""}
      ${plain.note ? `<p class="muted small">${esc(plain.note)}</p>` : ""}
    </div>`;
}

// Technical cards opened in the simple view stay open when a live run re-renders the page.
const openTechnical = new Set();

function technicalCard(title, body) {
  return simpleView
    ? `<details class="card technical" data-title="${esc(title)}" ${openTechnical.has(title) ? "open" : ""}><summary><h3>${esc(title)}</h3><span class="muted small">Technical details. Select to show.</span></summary>${body}</details>`
    : `<div class="card"><h3>${esc(title)}</h3>${body}</div>`;
}

$("#agent-detail").addEventListener("toggle", (e) => {
  const card = e.target.closest && e.target.closest("details.technical");
  if (!card) return;
  if (card.open) openTechnical.add(card.dataset.title); else openTechnical.delete(card.dataset.title);
}, true);

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
    ${plainCard(chosen.plain)}
    ${technicalCard("Report", `
      <div class="muted small">Recorded ${esc(fmtTime(chosen.recorded_at))}. Master Agent decision: ${esc(chosen.decision)}.</div>
      ${chosen.duration_seconds != null ? `<p class="muted small">Agent execution: ${esc(elapsedSeconds(chosen.duration_seconds))}. Started ${esc(fmtTime(chosen.started_at))}; finished ${esc(fmtTime(chosen.finished_at))}.</p>` : ""}
      <pre class="report">${esc(chosen.report_text || "No report text was recorded for this attempt.")}</pre>`)}
    ${technicalCard("Structured output", chosen.output == null
      ? `<div class="muted">No output was recorded for this attempt (the agent raised an error).</div>`
      : `<div class="json">${jsonTree(chosen.output)}</div>`)}`}`;
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
            ${vs ? `<span class="badge ${esc(vs)}">${esc(p.selection_source === "user_pdf" ? "User supplied" : p.validation_status)}</span>` : ""}
            ${p.has_code ? `<span class="badge">Has code</span>` : ""}
          </div>
        </div>
        <dl class="paper-body">
          <div><dt>Dataset</dt><dd>${esc(p.dataset || (p.selection_source === "user_pdf" ? "See supplied PDF" : "Not stated"))}</dd></div>
          <div><dt>Methodology</dt><dd>${esc(p.methodology || (p.selection_source === "user_pdf" ? "See supplied PDF" : "Not stated"))}</dd></div>
          <div><dt>Results</dt><dd>${esc(p.results_summary || (p.selection_source === "user_pdf" ? "See supplied PDF" : "Not stated"))}</dd></div>
          ${p.input_fingerprint ? `<div><dt>Fixed input</dt><dd>${esc(p.input_fingerprint.slice(0, 12))}</dd></div>` : ""}
          ${p.rejection_reason ? `<div><dt>Rejection reason</dt><dd>${esc(p.rejection_reason)}</dd></div>` : ""}
        </dl>
        ${p.abstract ? `<details><summary>Abstract</summary><p>${esc(p.abstract)}</p></details>` : ""}
        <div class="paper-actions">
          ${p.paper_url ? `<a class="btn sm" href="${esc(p.paper_url)}" target="_blank" rel="noopener noreferrer">Publisher page</a>` : ""}
          ${p.code_url ? `<a class="btn sm ghost" href="${esc(p.code_url)}" target="_blank" rel="noopener noreferrer">Code</a>` : ""}
          ${p.pdf_view_url || p.pdf_name
            ? `<button class="btn sm primary" data-view-pdf="${esc(p.pdf_view_url || `/api/pdfs/${p.pdf_name}/view`)}">View PDF</button>
               <a class="btn sm ghost" href="${esc(p.pdf_download_url || `/api/pdfs/${p.pdf_name}/download`)}">Download PDF</a>`
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
    $("#pdf-detail").innerHTML = `<div class="empty">This run has no documents yet. Fetch a paper's open-access PDF from the Papers page; the report and slides are on the Report &amp; slides page.</div>`;
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

// --- Report and slides ---

let reportData = null;
let reportTab = "report";
let deck = null;
let slideIndex = 0;

// A small Markdown renderer for the Report Agent's report.md: headings,
// paragraphs, tables, lists, quotes, links, figures and [P3] citations.
// Everything is escaped first, so the file's text can never become markup.
function mdInline(text, figures) {
  const held = [];
  const hold = (html) => `\u0000${held.push(html) - 1}\u0000`;
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, (_, c) => hold(`<code>${c}</code>`));
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => {
    const url = figures[src.split("/").pop()];
    return hold(url ? `<figure><img src="${esc(url)}" alt="${alt}"><figcaption>${alt}</figcaption></figure>`
      : `<span class="muted">[Figure not recorded by this run: ${alt}]</span>`);
  });
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, label, url) =>
    hold(`<a href="${url}" target="_blank" rel="noopener">${label}</a>`));
  s = s.replace(/https?:\/\/[^\s<>"]+/g, (url) => {
    const tail = (url.match(/[.,;:)]+$/) || [""])[0];
    const clean = url.slice(0, url.length - tail.length);
    return hold(`<a href="${clean}" target="_blank" rel="noopener">${clean}</a>`) + tail;
  });
  s = s.replace(/\[(P\d+)\]/g, (_, key) => hold(`<a class="cite" href="#" data-ref="${key}">[${key}]</a>`));
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:]|$)/g, "$1<em>$2</em>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => held[Number(i)]);
}

function mdCells(row) {
  return row.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "").split(/(?<!\\)\|/)
    .map((c) => c.trim().replace(/\\\|/g, "|"));
}

function renderMarkdown(md, figures) {
  const lines = md.replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  const toc = [];
  const isBlockStart = (l) => /^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|>|\s*\|)/.test(l);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if (!line.trim()) { i++; continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      const level = m[1].length;
      const id = `md-h${i}`;
      if (level === 2) toc.push({ id, text: m[2] });
      html.push(`<h${level} id="${id}">${mdInline(m[2], figures)}</h${level}>`);
      i++;
    } else if (/^\s*\|/.test(line) && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1] || "")) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cell = (c, tag) => `<${tag}${/^-?\d[\d.,]*%?$/.test(c) ? ' class="num"' : c.length > 120 ? ' class="long"' : ""}>${mdInline(c, figures)}</${tag}>`;
      html.push(`<div class="table-wrap md-table"><table class="table">
        <thead><tr>${mdCells(rows[0]).map((c) => cell(c, "th")).join("")}</tr></thead>
        <tbody>${rows.slice(2).map((r) => `<tr>${mdCells(r).map((c) => cell(c, "td")).join("")}</tr>`).join("")}</tbody>
      </table></div>`);
    } else if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const ordered = /^\s*\d+\.\s/.test(line);
      const items = [];
      while (i < lines.length && (ordered ? /^\s*\d+\.\s/ : /^\s*[-*]\s/).test(lines[i])) {
        const body = lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, "");
        const ref = body.match(/^\[(P\d+)\]/);
        items.push(`<li${ref ? ` id="ref-${ref[1]}"` : ""}>${mdInline(body, figures)}</li>`);
      }
      html.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
    } else if (line.startsWith(">")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith(">")) quote.push(lines[i++].replace(/^>\s?/, ""));
      html.push(`<blockquote>${mdInline(quote.join(" "), figures)}</blockquote>`);
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i]))) para.push(lines[i++]);
      html.push(`<p>${mdInline(para.join(" "), figures)}</p>`);
    }
  }
  return { html: html.join("\n"), toc };
}

async function loadReport() {
  try { reportData = await api(withRun("/api/report")); } catch (err) {
    $("#report-pane").innerHTML = `<div class="note err">${esc(err.message)}</div>`;
    return;
  }
  showReportTab(reportTab);
}

function showReportTab(tab) {
  reportTab = tab;
  document.querySelectorAll("#report-tabs .chip").forEach((c) => c.classList.toggle("active", c.dataset.tab === tab));
  $("#report-pane").hidden = tab !== "report";
  $("#slides-pane").hidden = tab !== "slides";
  if (tab === "report") renderReport(); else loadSlides();
}

function missingNote(what, agent) {
  const off = (reportData.turned_off || []).includes(agent);
  return `<div class="empty">${off ? `The ${esc(agent)} was turned off for this run, so it has no ${what}.`
    : `This run has no ${what} yet. The ${esc(agent)} makes it after the Evaluation and Experiment Agents finish.`}</div>`;
}

function outputMeta(item, extra = "") {
  const warnings = (item.warnings || []).map((w) => `<div class="note">${esc(w)}</div>`).join("");
  return `<div class="out-meta">
      ${item.status ? `<span class="badge ${esc(item.status.toLowerCase())}">${esc(item.status)}</span>` : ""}
      ${item.written_by ? `<span class="muted small">Text by ${esc(item.written_by)}</span>` : ""}
      ${extra}
      <span class="muted small mono">${esc(item.path)}</span>
      ${warnings}
    </div>`;
}

function renderReport() {
  const pane = $("#report-pane");
  const r = reportData && reportData.report;
  if (!r) { pane.innerHTML = missingNote("report", "Report Agent"); return; }
  const { html, toc } = renderMarkdown(r.text, r.figures || {});
  const stored = r.source === "database"
    ? `<span class="muted small">Shown from the run database: the file on disk has changed since this run.</span>` : "";
  pane.innerHTML = `
    <div class="card out-bar">
      ${outputMeta(r, `<span class="muted small">${r.removed_claims} sentence(s) removed by the fact check</span>${stored}`)}
      <div class="btn-group">
        ${r.facts_download_url ? `<a class="btn sm ghost" href="${esc(r.facts_download_url)}">Facts (JSON)</a>` : ""}
        <a class="btn sm primary" href="${esc(r.download_url)}">Download ${esc(r.name)}</a>
      </div>
    </div>
    <div class="split report-split">
      <nav class="list toc" aria-label="Report sections">
        ${toc.map((t) => `<button class="list-item" data-jump="${t.id}">${esc(t.text.replace(/\*\*/g, ""))}</button>`).join("")}
      </nav>
      <article class="card md-doc">
        ${r.truncated ? `<div class="note">The report is too long to show in full here. Download it to read the rest.</div>` : ""}
        ${html}
      </article>
    </div>`;
}

$("#report-tabs").addEventListener("click", (e) => {
  const chip = e.target.closest("[data-tab]");
  if (chip) showReportTab(chip.dataset.tab);
});

$("#report-pane").addEventListener("click", (e) => {
  const jump = e.target.closest("[data-jump]");
  const cite = e.target.closest("[data-ref]");
  if (!jump && !cite) return;
  e.preventDefault();  // the address bar holds the page and the run, not an anchor
  const target = document.getElementById(jump ? jump.dataset.jump : `ref-${cite.dataset.ref}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1400);
});

// Slides: each shape is placed where the .pptx puts it (fractions of the
// slide), and font sizes in points are scaled to the drawn slide's width,
// so the large view and the thumbnails use the same drawing.

const CHART_COLORS = ["#2A78D6", "#EB6834", "#1F8A4C", "#8A3FFC"];

function slideFont(pt) { return pt ? `font-size:${(pt / deck.width_pt * 100).toFixed(3)}cqw;` : ""; }

function slideParagraphs(paragraphs) {
  return paragraphs.map((p) => {
    const runs = p.runs.map((r) => `<span style="${slideFont(r.size)}${r.color ? `color:${r.color};` : ""}${
      r.bold ? "font-weight:700;" : ""}${r.italic ? "font-style:italic;" : ""}">${esc(r.text)}</span>`).join("");
    return `<p style="${p.align ? `text-align:${p.align};` : ""}${p.level ? `margin-left:${p.level * 4}cqw;` : ""}">${runs || "&nbsp;"}</p>`;
  }).join("");
}

function slideChart(shape) {
  const W = 600, H = 300, padL = 120, padR = 50, padT = shape.title ? 30 : 10, padB = shape.series.length > 1 ? 32 : 12;
  const cats = shape.categories, series = shape.series;
  const max = Math.max(...series.flatMap((s) => s.values).filter((v) => v != null), 0) || 1;  // the axis starts at zero
  const band = (H - padT - padB) / Math.max(cats.length, 1);
  const barH = (band * 0.7) / Math.max(series.length, 1);
  const x = (v) => padL + (v / max) * (W - padL - padR);
  const color = (s, si, ci) => (s.point_colors || [])[ci] || s.color || CHART_COLORS[si % CHART_COLORS.length];
  let out = shape.title ? `<text x="${W / 2}" y="18" text-anchor="middle" class="ct">${esc(shape.title)}</text>` : "";
  out += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" class="ax"/>`;
  cats.forEach((c, ci) => {
    const y0 = padT + ci * band + band * 0.15;
    out += `<text x="${padL - 8}" y="${y0 + band * 0.35}" text-anchor="end" dominant-baseline="middle" class="cl">${esc(c)}</text>`;
    series.forEach((s, si) => {
      const v = s.values[ci];
      if (v == null) return;
      const y = y0 + si * barH;
      out += `<rect x="${padL}" y="${y}" width="${Math.max(x(v) - padL, 0)}" height="${barH * 0.9}" fill="${color(s, si, ci)}">
        <title>${esc(s.name)}, ${esc(c)}: ${v}</title></rect>
        <text x="${x(v) + 4}" y="${y + barH * 0.45}" dominant-baseline="middle" class="dl">${v}</text>`;
    });
  });
  if (series.length > 1) {
    out += series.map((s, si) => `<rect x="${padL + si * 160}" y="${H - 16}" width="10" height="10" fill="${s.color || CHART_COLORS[si % CHART_COLORS.length]}"/>
      <text x="${padL + si * 160 + 14}" y="${H - 7}" class="cl">${esc(s.name)}</text>`).join("");
  }
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="slide-chart" role="img" aria-label="${esc(shape.title || "Chart")}">${out}</svg>`;
}

function slideShape(shape) {
  const box = `left:${shape.x * 100}%;top:${shape.y * 100}%;width:${shape.w * 100}%;height:${shape.h * 100}%;`;
  if (shape.type === "table") {
    const cols = shape.columns.map((w) => `<col style="width:${w * 100}%">`).join("");
    const rows = shape.rows.map((row) => `<tr>${row.map((c) =>
      `<td style="${c.fill ? `background:${c.fill};` : ""}">${slideParagraphs(c.paragraphs)}</td>`).join("")}</tr>`).join("");
    return `<div class="s-shape s-table" style="${box}"><table><colgroup>${cols}</colgroup>${rows}</table></div>`;
  }
  if (shape.type === "chart") return `<div class="s-shape" style="${box}">${slideChart(shape)}</div>`;
  let outline = "";
  if (shape.geometry === "pentagon") {  // PowerPoint's arrow-tip pentagon: the tip is half the height deep
    const depth = Math.min(50, (0.5 * shape.h * deck.height_pt) / (shape.w * deck.width_pt) * 100);
    outline = `clip-path:polygon(0 0,${100 - depth}% 0,100% 50%,${100 - depth}% 100%,0 100%);`;
  } else if (shape.geometry === "oval") outline = "border-radius:50%;";
  else if ((shape.geometry || "").startsWith("rounded")) outline = "border-radius:8%;";
  const anchor = { middle: "center", bottom: "flex-end" }[shape.anchor] || "flex-start";
  return `<div class="s-shape s-text" style="${box}${shape.fill ? `background:${shape.fill};` : ""}${outline}justify-content:${anchor};">${
    slideParagraphs(shape.paragraphs)}</div>`;
}

function slideHtml(slide) {
  return `<div class="slide" style="aspect-ratio:${deck.width_pt} / ${deck.height_pt}">${slide.shapes.map(slideShape).join("")}</div>`;
}

async function loadSlides() {
  const pane = $("#slides-pane");
  const p = reportData && reportData.presentation;
  if (!p) { pane.innerHTML = missingNote("slides", "Presentation Agent"); return; }
  if (!deck || deck.url !== p.slides_url) {
    pane.innerHTML = `<div class="muted small">Reading the slides…</div>`;
    try { deck = { ...(await api(p.slides_url)), url: p.slides_url }; } catch (err) {
      deck = null;
      pane.innerHTML = `<div class="note err">${esc(err.message)}</div>
        <a class="btn sm primary" href="${esc(p.download_url)}">Download ${esc(p.name)}</a>`;
      return;
    }
    slideIndex = Math.min(slideIndex, deck.slides.length - 1);
  }
  const fallback = (p.fallback_slides || []).length
    ? `<span class="muted small">${p.fallback_slides.length} slide(s) use the report's own sentences</span>` : "";
  pane.innerHTML = `
    <div class="card out-bar">
      ${outputMeta(p, fallback)}
      <div class="btn-group">
        <button class="btn sm ghost" id="slide-full">Full screen</button>
        <a class="btn sm primary" href="${esc(p.download_url)}">Download ${esc(p.name)}</a>
      </div>
    </div>
    <div class="deck">
      <div class="deck-main">
        <div class="slide-stage" id="slide-stage"></div>
        <div class="deck-nav">
          <button class="btn sm" id="slide-prev">Previous</button>
          <span class="muted small" id="slide-count"></span>
          <button class="btn sm" id="slide-next">Next</button>
        </div>
        <div class="card slide-notes"><div class="label">Speaker notes</div><div id="slide-notes" class="muted small"></div></div>
      </div>
      <div class="thumbs" id="slide-thumbs">
        ${deck.slides.map((s, i) => `<button class="thumb" data-slide="${i}" aria-label="Slide ${s.number}: ${esc(s.title)}">
          ${slideHtml(s)}<span class="thumb-label">${s.number}. ${esc(s.title)}</span></button>`).join("")}
      </div>
    </div>`;
  showSlide(slideIndex);
}

function showSlide(i) {
  if (!deck || !$("#slide-stage")) return;
  slideIndex = Math.max(0, Math.min(i, deck.slides.length - 1));
  const slide = deck.slides[slideIndex];
  $("#slide-stage").innerHTML = slideHtml(slide);
  $("#slide-count").textContent = `Slide ${slideIndex + 1} of ${deck.slides.length}: ${slide.title}`;
  $("#slide-notes").textContent = slide.notes || "No speaker notes.";
  $("#slide-prev").disabled = slideIndex === 0;
  $("#slide-next").disabled = slideIndex === deck.slides.length - 1;
  document.querySelectorAll("#slide-thumbs .thumb").forEach((t) => {
    const on = Number(t.dataset.slide) === slideIndex;
    t.classList.toggle("active", on);
    if (on) t.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}

$("#slides-pane").addEventListener("click", (e) => {
  const thumb = e.target.closest(".thumb");
  if (e.target.closest("#slide-prev")) showSlide(slideIndex - 1);
  else if (e.target.closest("#slide-next")) showSlide(slideIndex + 1);
  else if (e.target.closest("#slide-full")) { const stage = $("#slide-stage"); if (stage.requestFullscreen) stage.requestFullscreen(); }
  else if (thumb) showSlide(Number(thumb.dataset.slide));
});

document.addEventListener("keydown", (e) => {
  if (currentView !== "report" || reportTab !== "slides" || !deck) return;
  if (e.target.closest && e.target.closest("input, textarea, select")) return;
  if (["ArrowRight", "PageDown", " "].includes(e.key)) { e.preventDefault(); showSlide(slideIndex + 1); }
  else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); showSlide(slideIndex - 1); }
  else if (e.key === "Home") showSlide(0);
  else if (e.key === "End") showSlide(deck.slides.length - 1);
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

// --- Compare runs (read-only; the same numbers as `run_metrics compare`) ---

const fmtVal = (v) => (v == null ? "—" : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toPrecision(4)) : esc(v));
let compareFieldsLoaded = false;

async function loadCompare() {
  const params = new URLSearchParams({
    group_by: $("#cmp-group-by").value || "memory_mode", topic: $("#cmp-topic").value.trim(),
    group: $("#cmp-group").value, all_tasks: $("#cmp-all-tasks").checked ? "1" : "",
  });
  let data;
  try { data = await api(`/api/compare?${params}`); } catch (err) {
    $("#cmp-blocks").innerHTML = `<div class="empty">${esc(err.message)}</div>`; return;
  }
  if (!compareFieldsLoaded) {
    $("#cmp-group-by").innerHTML = data.group_fields.map((f) =>
      `<option value="${esc(f)}" ${f === data.group_by ? "selected" : ""}>${esc(f.replace(/_/g, " "))}</option>`).join("");
    $("#cmp-group").innerHTML = `<option value="">All runs</option>` +
      data.groups_available.map((g) => `<option value="${esc(g)}">${esc(g.replace(/^test:/, ""))}</option>`).join("");
    compareFieldsLoaded = true;
  }
  $("#cmp-kind").textContent = data.kind === "controlled"
    ? "A controlled test: runs grouped by the condition they were assigned before they started."
    : "Descriptive: these runs were not a planned test, so differences cannot show what caused them.";
  const shown = data.runs.length ? Object.keys(data.runs[0].values) : [];
  $("#cmp-runs").innerHTML = data.runs.length ? `
    <thead><tr><th>Run</th><th>Status</th><th>${esc(data.group_by.replace(/_/g, " "))}</th><th>Memory exposure</th>
      ${shown.map((m) => `<th>${esc(m.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
    <tbody>${data.runs.map((r) => `<tr>
      <td><a href="${hashFor("overview", r.run_id)}">${esc(r.run_id)}</a><div class="muted small">${esc(r.topic || "")}</div></td>
      <td><span class="badge ${esc(r.status)}">${esc(r.status)}</span></td><td>${esc(r.group ?? "—")}</td>
      <td>${esc(r.memory_exposure ?? "—")}</td>${shown.map((m) => `<td>${fmtVal(r.values[m])}</td>`).join("")}</tr>`).join("")}</tbody>`
    : `<tbody><tr><td class="muted">No measured runs match.</td></tr></tbody>`;
  $("#cmp-blocks").innerHTML = data.blocks.map((b) => `
    <div class="card cmp-block"><div style="padding:14px">
      <h2>Task: ${esc(b.task)} <span class="muted small">(${b.runs} run(s))</span></h2>
      <div class="table-wrap"><table class="table">
        <thead><tr><th>Metric</th>${Object.entries(b.groups).map(([g, info]) =>
          `<th>${esc(g)} <span class="muted small">${info.runs} run(s)${info.too_few_runs ? ", too few runs: no conclusion" : ""}</span></th>`).join("")}</tr></thead>
        <tbody>${data.metrics.map((m) => `<tr><td>${esc(m.replace(/_/g, " "))}</td>${Object.values(b.groups).map((info) => {
          const s = info.metrics[m];
          return `<td>${s.n ? `median ${fmtVal(s.median)} <span class="muted small">mean ${fmtVal(s.mean)}, range ${fmtVal(s.min)} to ${fmtVal(s.max)}, n=${s.n}/${s.runs}</span>` : `<span class="muted">n=0/${s.runs}</span>`}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table></div>
      ${Object.keys(b.tests).length ? `<p class="small"><strong>Tests</strong> (two-sided permutation test on the mean; Cliff's delta, first group vs second): ${
        Object.entries(b.tests).map(([m, t]) => `${esc(m)}: p=${t.p_value.toFixed(4)}, delta=${t.cliffs_delta.toFixed(2)}, n=${t.n.join("/")}`).join("; ")}</p>`
        : Object.keys(b.groups).length === 2 ? `<p class="small muted">No test: each group needs at least 3 runs with a value.</p>` : ""}
      ${Object.keys(b.confounders).length ? `<p class="small cmp-note"><strong>Conditions that differ</strong> (possible confounders): ${
        Object.entries(b.confounders).map(([f, v]) => `${esc(f)}: ${Object.entries(v).map(([g, vals]) => `${esc(g)} = ${esc(vals.join(", ").slice(0, 120))}`).join("; ")}`).join(" | ")}</p>` : ""}
    </div></div>`).join("") || `<div class="empty">Nothing to compare.</div>`;
}

["#cmp-group-by", "#cmp-group", "#cmp-all-tasks"].forEach((s) => $(s).addEventListener("change", loadCompare));
$("#cmp-topic").addEventListener("change", loadCompare);

// --- Memory (read-only) ---

let memoryAgentsLoaded = false;
let memorySelected = null;

async function loadMemory(refresh = false) {
  const params = new URLSearchParams({ agent: $("#mem-agent").value, state: $("#mem-state").value,
                                       q: $("#mem-search").value.trim(), refresh: refresh ? "1" : "" });
  let data;
  try { data = await api(`/api/memories?${params}`); } catch (err) {
    $("#mem-table").innerHTML = `<tbody><tr><td>${esc(err.message)}</td></tr></tbody>`; return;
  }
  if (!memoryAgentsLoaded) {
    $("#mem-agent").innerHTML = `<option value="">All agents</option>` +
      data.agents.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join("");
    memoryAgentsLoaded = true;
  }
  const states = Object.entries(data.states).map(([s, n]) => `${n} ${s}`).join(", ");
  $("#mem-source").textContent = `From ${data.source}: ${data.total} memories (${states}). Showing ${data.memories.length}.`;
  const best = new Set(Object.values(data.by_agent).flatMap((a) => a.best));
  const worst = new Set(Object.values(data.by_agent).flatMap((a) => a.worst));
  $("#mem-table").innerHTML = data.memories.length ? `
    <thead><tr><th>Score</th><th>Agent</th><th>Kind</th><th>Title</th></tr></thead>
    <tbody>${data.memories.map((m) => `<tr data-memory="${esc(m.id)}" class="${m.id === memorySelected ? "selected" : ""}">
      <td><strong>${m.effective_quality.toFixed(2)}</strong>${best.has(m.id) ? ` <span class="badge ok">best</span>` : ""}${worst.has(m.id) ? ` <span class="badge failed">worst</span>` : ""}</td>
      <td>${esc(m.agent)}</td><td><span class="badge ${esc(m.polarity)}">${esc(m.kind)}</span></td>
      <td>${esc(m.title)}<div class="muted small">lineage ${esc(m.lineage)}, ${esc(m.state)}${m.times_helped || m.times_hurt ? `, helped ${m.times_helped}, hurt ${m.times_hurt}` : ""}</div></td></tr>`).join("")}</tbody>`
    : `<tbody><tr><td class="muted">No memories match.</td></tr></tbody>`;
}

async function showMemory(id) {
  memorySelected = id;
  document.querySelectorAll("#mem-table tr[data-memory]").forEach((tr) => tr.classList.toggle("selected", tr.dataset.memory === id));
  let m;
  try { m = await api(`/api/memories/${encodeURIComponent(id)}`); } catch (err) {
    $("#mem-detail").innerHTML = `<div class="empty">${esc(err.message)}</div>`; return;
  }
  const reasons = Object.entries(m.quality_reasons || {}).filter(([, v]) => v && typeof v === "object" && "value" in v);
  $("#mem-detail").innerHTML = `
    <h2 style="margin-top:0;font-size:15px">${esc(m.title)}</h2>
    <p class="small"><span class="badge ${esc(m.polarity)}">${esc(m.kind)} / ${esc(m.polarity)}</span>
      <span class="badge ${esc(m.state)}">${esc(m.state)}</span> lineage ${esc(m.lineage)}, trust ${esc(m.trust)}, ${esc(m.agent)}</p>
    <p>${esc(m.summary)}</p>
    <p class="small"><strong>Effective quality ${m.effective_quality.toFixed(2)}</strong> (base ${Number(m.quality || 0).toFixed(2)};
      helped ${m.times_helped}, hurt ${m.times_hurt}). ${esc((m.quality_reasons || {}).formula || "")}</p>
    ${reasons.length ? `<table class="table small"><tbody>${reasons.map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${fmtVal(v.value)}</td><td>${esc(v.reason || "")}</td></tr>`).join("")}</tbody></table>` : ""}
    <p class="small muted">Shown to agents ${m.use.shown} time(s), used ${m.use.used} (the agents' own report), followed by
      ${m.use.helped} good and ${m.use.hurt} bad outcome(s). From run ${esc(m.run_id || "none")}; source: ${esc(m.source)}.</p>
    ${m.state_changes.length ? `<p class="small"><strong>State changes</strong></p><ul class="small">${m.state_changes.map((c) =>
      `<li>${esc(fmtTime(c.created_at))}: ${esc(c.old_state || "none")} to ${esc(c.new_state)} by ${esc(c.actor || "unknown")}: ${esc(c.reason || "")}</li>`).join("")}</ul>` : ""}
    <p class="small"><strong>Task features</strong></p><pre>${esc(JSON.stringify(m.task || {}, null, 1))}</pre>
    <p class="small"><strong>Evidence</strong></p><pre>${esc(JSON.stringify(m.evidence || {}, null, 1))}</pre>
    ${m.content ? `<p class="small"><strong>Content</strong></p><pre>${esc(m.content)}</pre>` : ""}`;
}

$("#mem-table").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-memory]");
  if (row) showMemory(row.dataset.memory);
});
["#mem-agent", "#mem-state"].forEach((s) => $(s).addEventListener("change", () => loadMemory()));
$("#mem-search").addEventListener("change", () => loadMemory());
$("#mem-refresh").addEventListener("click", () => loadMemory(true));

// --- Start ---

// --- Optional agents: on/off switches for the next run ---

// Optional agents' switches (shown on their pipeline cards), by agent name.
let switches = {};

function setSwitches(agents) {
  switches = Object.fromEntries((agents || []).map((a) => [a.name, a]));
}

async function loadSwitches() {
  try { setSwitches((await api("/api/agent-switches")).agents); } catch { switches = {}; }
  if (status) renderPipeline();
}

loadSwitches();

async function tick() {
  await pollStatus();
  if (currentView === "overview") await pollConsole();
}

route();
tick();
setInterval(tick, 2000);
