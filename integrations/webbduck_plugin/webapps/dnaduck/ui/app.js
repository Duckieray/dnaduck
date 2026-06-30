(function () {
  const IS_PLUGIN = window.location.pathname.includes("/plugins/");
  const API_BASE = IS_PLUGIN
    ? window.location.pathname.replace(/\/ui\/.*$/, "/api")
    : window.location.pathname.replace(/\/ui\/.*$/, "") || "/";
  const IDENTITY_PAGE_SIZE = 120;
  const ACTIVITY_POLL_MS_ACTIVE = 3000;
  const ACTIVITY_POLL_MS_IDLE = 15000;
  const EVENT_FEED_LIMIT = 8;

  let activityPollInFlight = false;
  let activityTimer = null;
  let identitiesCache = [];
  let selectedIdentityId = null;
  let selectedIdentityOffset = 0;
  let selectedIdentityTotal = 0;
  let selectedIdentityRows = [];
  let selectedExportIdentityIds = new Set();
  let eventFeed = [];
  let lastActivityNoticeKey = null;
  let pausedJobsCache = [];

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, text) {
    const node = byId(id);
    if (!node) return;
    node.textContent = String(text || "");
  }

  function normalizeOptionalPath(value) {
    let raw = String(value || "").trim();
    if (!raw) return "";
    if (raw.length >= 2 && raw[0] === raw[raw.length - 1] && (raw[0] === "\"" || raw[0] === "'")) {
      raw = raw.slice(1, -1).trim();
    }
    return raw;
  }

  function numberText(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "0";
    return n.toLocaleString();
  }

  function firstNonEmptyLine(text) {
    const raw = String(text || "");
    const lines = raw.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    return lines.length ? lines[0] : "";
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds || 0)));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins <= 0) return `${secs}s`;
    return `${mins}m ${secs.toString().padStart(2, "0")}s`;
  }

  function formatDateTime(epochSeconds) {
    const num = Number(epochSeconds);
    if (!Number.isFinite(num) || num <= 0) return "unknown time";
    return new Date(num * 1000).toLocaleString();
  }

  function nowStamp() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function addEvent(text) {
    const clean = String(text || "").trim();
    if (!clean) return;
    eventFeed.unshift(`[${nowStamp()}] ${clean}`);
    eventFeed = eventFeed.slice(0, EVENT_FEED_LIMIT);
    renderEventFeed();
  }

  function renderEventFeed() {
    const feed = byId("event-feed");
    if (!feed) return;
    feed.innerHTML = "";
    if (!eventFeed.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No updates yet.";
      feed.appendChild(li);
      return;
    }
    for (const row of eventFeed) {
      const li = document.createElement("li");
      li.textContent = row;
      feed.appendChild(li);
    }
  }

  function setButtonBusy(buttonId, busy, busyText) {
    const btn = byId(buttonId);
    if (!btn) return;
    if (!btn.dataset.defaultText) {
      btn.dataset.defaultText = btn.textContent || "";
    }
    btn.disabled = !!busy;
    btn.textContent = busy ? busyText : btn.dataset.defaultText;
  }

  function parseErrorDetail(payload) {
    if (payload == null) return "Something went wrong.";
    if (Array.isArray(payload)) {
      if (!payload.length) return "Something went wrong.";
      const first = payload[0];
      if (typeof first === "string") return first;
      if (first && typeof first === "object" && typeof first.msg === "string") return first.msg;
      try {
        return JSON.stringify(first);
      } catch (_ignored) {
        return "Something went wrong.";
      }
    }
    if (typeof payload === "string") {
      const trimmed = payload.trim();
      if (!trimmed) return "Something went wrong.";
      try {
        return parseErrorDetail(JSON.parse(trimmed));
      } catch (_ignored) {
        return trimmed;
      }
    }
    if (typeof payload === "object") {
      const detail = payload.detail;
      if (typeof detail === "string") return detail;
      if (detail && typeof detail === "object") {
        if (
          typeof detail.error === "string" &&
          detail.error.toLowerCase().includes("remote dnaduck api request failed") &&
          typeof detail.body === "string" &&
          detail.body.trim()
        ) {
          const bodyText = detail.body.trim();
          try {
            return parseErrorDetail(JSON.parse(bodyText));
          } catch (_ignored) {
            return bodyText;
          }
        }
        if (typeof detail.error === "string") return detail.error;
        if (typeof detail.message === "string") return detail.message;
        if (typeof detail.body === "string" && detail.body.trim()) return detail.body.trim();
      }
      if (typeof payload.error === "string") return payload.error;
      if (typeof payload.message === "string") return payload.message;
      try {
        return JSON.stringify(payload);
      } catch (_ignored) {
        return "Something went wrong.";
      }
    }
    return "Something went wrong.";
  }

  async function request(path, options) {
    const response = await fetch(`${API_BASE}${path}`, options || {});
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      throw new Error(parseErrorDetail(payload));
    }
    return payload;
  }

  function get(path) {
    return request(path, { method: "GET" });
  }

  function post(path, body) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  function put(path, body) {
    return request(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  }

  function setStatValues(values) {
    if (!values || typeof values !== "object") return;
    if (values.faces_found != null) setText("stat-faces-found", numberText(values.faces_found));
    if (values.new_processed != null) setText("stat-new-processed", numberText(values.new_processed));
    if (values.identities != null) setText("stat-identities", numberText(values.identities));
    if (values.review_count != null) setText("stat-review-count", numberText(values.review_count));
  }

  function selectedIdsSorted() {
    return Array.from(selectedExportIdentityIds)
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0)
      .sort((a, b) => a - b);
  }

  function getSelectedIdsForExport() {
    const useSelected = Boolean(byId("use-selected-only")?.checked);
    if (!useSelected) return null;
    return selectedIdsSorted();
  }

  function updateSelectedExportSummary() {
    const node = byId("selected-export-summary");
    if (!node) return;
    const useSelected = Boolean(byId("use-selected-only")?.checked);
    const selected = selectedIdsSorted();
    if (useSelected) {
      if (!selected.length) {
        node.textContent = "Selection mode is on, but no groups are selected yet. Choose groups in Characters first.";
      } else {
        node.textContent = `${numberText(selected.length)} character groups selected for export and training.`;
      }
      return;
    }
    if (!selected.length) {
      node.textContent = "Selection mode is off. DNADuck will use all groups that meet Minimum Photos.";
      return;
    }
    node.textContent =
      `Selection mode is off. DNADuck will use all groups that meet Minimum Photos (${numberText(selected.length)} selected groups currently ignored).`;
  }

  function scanPayloadFromResponse(result) {
    if (!result || typeof result !== "object") return {};
    if (result.result && typeof result.result === "object") return result.result;
    return result;
  }

  function exportPayloadFromResponse(result) {
    if (!result || typeof result !== "object") return {};
    if (result.result && typeof result.result === "object") return result.result;
    return result;
  }

  function trainPayloadFromResponse(result) {
    if (!result || typeof result !== "object") return {};
    if (result.result && typeof result.result === "object") return result.result;
    return result;
  }

  function compactIdentityList(identityIds) {
    const ids = Array.isArray(identityIds)
      ? identityIds.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0)
      : [];
    if (!ids.length) return "all eligible groups";
    if (ids.length <= 6) return `groups ${ids.join(", ")}`;
    const preview = ids.slice(0, 6).join(", ");
    return `groups ${preview} (+${ids.length - 6} more)`;
  }

  function pausedJobLabel(job) {
    const request = job && typeof job.request_summary === "object" ? job.request_summary : {};
    const identityIds = Array.isArray(request.identity_ids) ? request.identity_ids : [];
    const status = String(job?.status || "paused");
    const createdAt = formatDateTime(job?.created_at);
    const identityText = compactIdentityList(identityIds);
    const shortJobId = String(job?.job_id || "").slice(0, 8) || "unknown";
    return `${shortJobId} | ${status} | ${identityText} | ${createdAt}`;
  }

  function pausedJobSummary(job) {
    if (!job || typeof job !== "object") {
      return "Paused runs remember their character selection and training settings.";
    }
    const request = job && typeof job.request_summary === "object" ? job.request_summary : {};
    const result = job && typeof job.result_summary === "object" ? job.result_summary : {};
    const identityIds = Array.isArray(request.identity_ids) ? request.identity_ids : [];
    const parts = [];
    parts.push(`Created: ${formatDateTime(job.created_at)}`);
    parts.push(`Characters: ${compactIdentityList(identityIds)}`);
    if (request.min_images != null) parts.push(`Min photos: ${request.min_images}`);
    if (request.output_folder) parts.push(`Output: ${request.output_folder}`);
    if (result.output_dir) parts.push(`Trainer output: ${result.output_dir}`);
    return parts.join(" | ");
  }

  function getSelectedPausedJob() {
    const selectedId = String(byId("paused-job-select")?.value || "").trim();
    if (!selectedId) return null;
    return pausedJobsCache.find((job) => String(job?.job_id || "").trim() === selectedId) || null;
  }

  function updatePausedJobSummary() {
    setText("paused-job-summary", pausedJobSummary(getSelectedPausedJob()));
  }

  function renderPausedJobs(jobs, preferredJobId) {
    const select = byId("paused-job-select");
    if (!select) return;
    pausedJobsCache = Array.isArray(jobs) ? jobs : [];
    const preferred = String(preferredJobId || select.value || "").trim();
    select.innerHTML = "";

    if (!pausedJobsCache.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No paused runs found.";
      select.appendChild(option);
      select.disabled = true;
      updatePausedJobSummary();
      return;
    }

    for (const job of pausedJobsCache) {
      const option = document.createElement("option");
      option.value = String(job.job_id || "");
      option.textContent = pausedJobLabel(job);
      select.appendChild(option);
    }
    select.disabled = false;
    if (preferred) {
      const exists = pausedJobsCache.some((job) => String(job.job_id || "") === preferred);
      if (exists) select.value = preferred;
    }
    if (!select.value && pausedJobsCache[0]) {
      select.value = String(pausedJobsCache[0].job_id || "");
    }
    updatePausedJobSummary();
  }

  async function loadPausedJobs(preferredJobId) {
    try {
      const response = await get("/train-jobs?status=paused&limit=100");
      const jobs = Array.isArray(response?.jobs) ? response.jobs : [];
      renderPausedJobs(jobs, preferredJobId);
      return jobs;
    } catch (error) {
      const select = byId("paused-job-select");
      if (select) {
        select.innerHTML = "";
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "Paused runs unavailable in this mode.";
        select.appendChild(option);
        select.disabled = true;
      }
      pausedJobsCache = [];
      setText("paused-job-summary", `Could not load paused runs: ${error.message}`);
      return [];
    }
  }

  function renderActivity(activity) {
    const dot = byId("activity-dot");
    const state = byId("activity-state");
    const message = byId("activity-message");
    const elapsed = byId("activity-elapsed");
    const progress = byId("activity-progress");
    const bar = byId("activity-progress-bar");
    if (!dot || !state || !message || !elapsed || !progress || !bar) return;

    const payload = activity && typeof activity === "object" ? activity : {};
    const remote = payload.remote_activity && typeof payload.remote_activity === "object"
      ? payload.remote_activity
      : null;
    const active = remote || payload;
    const running = (remote && typeof remote.running === "boolean")
      ? Boolean(remote.running)
      : Boolean(payload.running);
    const operation = String(active.operation || payload.operation || "").trim();
    const stage = String(active.stage || payload.stage || "").trim();
    const lastError = String(active.last_error || payload.last_error || "").trim();
    const rawMessage = String(active.message || payload.message || "").trim();
    const isPausedState = !running && operation === "train_lora" && /paused/i.test(rawMessage);

    dot.classList.remove("running", "error");
    if (running) {
      dot.classList.add("running");
      if (operation === "scan_recluster") state.textContent = "Starting Fresh Scan";
      else if (operation === "scan") state.textContent = "Finding Similar Faces";
      else if (operation === "export_lora") state.textContent = "Preparing Training Set";
      else if (operation === "train_lora") state.textContent = "Training";
      else state.textContent = "Working";
      if (stage) state.textContent += ` (${stage})`;
    } else if (lastError) {
      dot.classList.add("error");
      state.textContent = "Needs Attention";
    } else if (isPausedState) {
      state.textContent = "Paused";
    } else {
      state.textContent = "Ready";
    }

    const msg = rawMessage;
    message.textContent = msg || (running ? "Working..." : "No active task.");
    if (running && (operation === "scan" || operation === "scan_recluster")) {
      message.textContent += " First run may download model files.";
    }

    const elapsedS = active.elapsed_s ?? payload.elapsed_s;
    const lastDurationS = active.last_duration_s ?? payload.last_duration_s;
    if (running && elapsedS != null) {
      elapsed.textContent = `Elapsed: ${formatDuration(elapsedS)}`;
    } else if (!running && lastDurationS != null) {
      elapsed.textContent = `Last run: ${formatDuration(lastDurationS)}`;
    } else {
      elapsed.textContent = "";
    }

    const processed = Number(active.processed_count);
    const total = Number(active.total_count);
    const discovered = Number(active.discovered_count);
    const toProcess = Number(active.total_to_process);
    const assigned = Number(active.assigned_count);
    const noise = Number(active.noise_count);
    const noFace = Number(active.no_face_count);
    const etaS = Number(active.eta_s);
    const progressPct = Number(active.progress_pct);

    let barWidth = 0;
    if (Number.isFinite(processed) && Number.isFinite(total) && total > 0) {
      barWidth = Math.max(0, Math.min(100, (processed / total) * 100));
      if (operation === "train_lora") {
        const pctText = Number.isFinite(progressPct)
          ? `${Math.max(0, Math.min(100, progressPct)).toFixed(1)}%`
          : `${((processed / total) * 100).toFixed(1)}%`;
        const etaText = Number.isFinite(etaS) && etaS >= 0 ? ` | ETA: ${formatDuration(etaS)}` : "";
        progress.textContent = `Progress: step ${numberText(processed)} / ${numberText(total)} (${pctText})${etaText}`;
      } else {
        progress.textContent = `Progress: ${numberText(processed)} of ${numberText(total)} images`;
      }
    } else if (Number.isFinite(discovered) && discovered > 0 && Number.isFinite(toProcess) && toProcess >= 0) {
      const checked = Math.max(0, discovered - toProcess);
      barWidth = Math.max(0, Math.min(100, (checked / discovered) * 100));
      progress.textContent = `Checked ${numberText(checked)} of ${numberText(discovered)} files`;
    } else if (running) {
      barWidth = 20;
      progress.textContent = operation === "train_lora" ? "Training in progress..." : "Preparing files...";
    } else {
      progress.textContent = "No active work.";
    }
    bar.style.width = `${barWidth}%`;

    const reviewCount = (Number.isFinite(noise) ? noise : 0) + (Number.isFinite(noFace) ? noFace : 0);
    setStatValues({
      faces_found: Number.isFinite(discovered) ? discovered : null,
      new_processed: Number.isFinite(processed) ? processed : null,
      identities: Number.isFinite(active.identity_count) ? Number(active.identity_count) : null,
      review_count: reviewCount,
    });

    const completedAt = active.last_completed_at ?? payload.last_completed_at;

    if (!running && lastError) {
      setText("last-action-text", `Could not finish: ${lastError}`);
      const errorKey = `error:${operation}:${String(completedAt || "")}:${lastError}`;
      if (lastActivityNoticeKey !== errorKey) {
        addEvent(`Action failed: ${lastError}`);
        lastActivityNoticeKey = errorKey;
      }
    } else if (!running && stage === "complete") {
      setText("last-action-text", "Latest action completed successfully.");
    }

    if (Number.isFinite(assigned) || Number.isFinite(noise) || Number.isFinite(noFace)) {
      const grouped = Number.isFinite(assigned) ? assigned : 0;
      const review = (Number.isFinite(noise) ? noise : 0) + (Number.isFinite(noFace) ? noFace : 0);
      setStatValues({ review_count: review, new_processed: Number.isFinite(processed) ? processed : null });
      if (!running) {
        const completeKey = `complete:${operation}:${String(completedAt || "")}:${grouped}:${review}`;
        if (lastActivityNoticeKey !== completeKey) {
          addEvent(`Grouped ${numberText(grouped)} images. ${numberText(review)} need review.`);
          lastActivityNoticeKey = completeKey;
        }
      }
    }
  }

  function isActivityRunning(activity) {
    const payload = activity && typeof activity === "object" ? activity : {};
    const remote = payload.remote_activity && typeof payload.remote_activity === "object"
      ? payload.remote_activity
      : null;
    if (remote && typeof remote.running === "boolean") {
      return Boolean(remote.running);
    }
    return Boolean(payload.running);
  }

  function scheduleActivityPoll(delayMs) {
    if (activityTimer) {
      window.clearTimeout(activityTimer);
      activityTimer = null;
    }
    const delay = Math.max(1000, Number(delayMs) || ACTIVITY_POLL_MS_IDLE);
    activityTimer = window.setTimeout(() => {
      void pollActivityAndReschedule();
    }, delay);
  }

  async function pollActivityAndReschedule() {
    const activity = await pollActivity();
    if (isActivityRunning(activity)) {
      scheduleActivityPoll(ACTIVITY_POLL_MS_ACTIVE);
      return;
    }
    scheduleActivityPoll(ACTIVITY_POLL_MS_IDLE);
  }

  async function pollActivity() {
    if (activityPollInFlight) return null;
    activityPollInFlight = true;
    try {
      const activity = await get("/activity");
      renderActivity(activity);
      return activity;
    } catch (error) {
      const fallback = {
        running: false,
        operation: null,
        message: "Could not read activity right now.",
        last_error: String(error?.message || "activity_unavailable"),
      };
      renderActivity(fallback);
      return fallback;
    } finally {
      activityPollInFlight = false;
    }
  }

  function setConnectionInfo(health) {
    const infoNode = byId("connection-info");
    const indicator = byId("status-indicator");
    const summaryNode = byId("connection-summary");
    if (!infoNode || !indicator) return;

    const mode = String(health?._plugin_mode || "unknown");
    const remoteBase = String(health?._remote_api_base || "").trim();
    indicator.classList.remove("ok", "warn", "error");

    if (mode === "remote_api") {
      indicator.classList.add("ok");
      infoNode.textContent = "Connected";
      if (summaryNode) summaryNode.textContent = `Remote connection active (${remoteBase || "custom host"}).`;
      return;
    }

    if (mode === "managed_api") {
      indicator.classList.add("ok");
      infoNode.textContent = "Connected";
      if (summaryNode) summaryNode.textContent = "Built-in DNADuck service is active.";
      return;
    }

    if (mode === "local_cli") {
      indicator.classList.add("warn");
      infoNode.textContent = "Local Mode";
      if (summaryNode) summaryNode.textContent = "Running through local command mode.";
      return;
    }

    indicator.classList.add("error");
    infoNode.textContent = "Disconnected";
    if (summaryNode) summaryNode.textContent = "Connection not ready. Check plugin settings.";
  }

  function switchView(nextView) {
    const view = String(nextView || "studio").trim();
    document.querySelectorAll(".nav-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === view);
    });
    document.querySelectorAll(".view").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `view-${view}`);
    });
  }

  async function loadHealth() {
    const health = await get("/health");
    setConnectionInfo(health);
    addEvent("Connection checked.");
    return health;
  }

  // ── Config management ─────────────────────────────────────────────

  let configsCache = [];
  let configNamesCache = [];
  let currentConfigName = "";
  let currentConfig = {};

  async function loadConfigList() {
    try {
      const data = await get("/configs");
      configsCache = Array.isArray(data.configs) ? data.configs : [];
      currentConfigName = String(data.current || "");
      renderConfigSelector(currentConfigName);
      return data;
    } catch (error) {
      return { configs: [], current: null };
    }
  }

  async function loadCurrentConfig() {
    try {
      currentConfig = await get("/config");
      renderConfigEditor(currentConfig);
      return currentConfig;
    } catch (error) {
      return {};
    }
  }

  function renderConfigSelector(currentName) {
    const select = byId("config-select");
    if (!select) return;
    select.innerHTML = "";
    for (const name of configsCache) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      if (name === currentName) opt.selected = true;
      select.appendChild(opt);
    }
    select.disabled = configsCache.length <= 1;
    const label = byId("cfg-current-config");
    if (label) label.textContent = currentName || "config.yaml";
  }

  function setField(id, value) {
    const el = byId(id);
    if (!el) return;
    if (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA") {
      el.value = String(value ?? "");
    } else {
      el.textContent = String(value ?? "");
    }
  }

  function renderConfigEditor(config) {
    if (!config || typeof config !== "object") return;
    setField("cfg-input-folder", config.input_folder || "");
    setField("cfg-output-folder", config.output_folder || "");
    setField("cfg-database-path", config.database_path || "");
    setField("cfg-mode", config.mode || "realism");
    setField("cfg-eps-realism", config.eps_realism ?? "");
    setField("cfg-min-samples", config.min_samples ?? "");
    setField("cfg-lora-min-images", config.lora_min_images ?? "");
    setField("cfg-train-steps", config.kohya_train_steps ?? "");
    setField("cfg-learning-rate", config.kohya_learning_rate ?? "");
    setField("cfg-network-dim", config.kohya_network_dim ?? "");
    setField("cfg-network-alpha", config.kohya_network_alpha ?? "");
    setField("cfg-batch-size", config.kohya_batch_size ?? "");
    setField("cfg-num-repeats", config.kohya_num_repeats ?? "");
    setField("cfg-base-model", config.kohya_base_model || "");
    setField("cfg-output-name", config.kohya_output_name || "");
    setField("cfg-current-config", currentConfigName || "config.yaml");
    // Environment variables
    const env = config.env && typeof config.env === "object" ? config.env : {};
    const envLines = Object.entries(env)
      .filter(([, v]) => v != null && String(v).trim() !== "")
      .map(([k, v]) => `${k}=${v}`)
      .join("\n");
    const envEl = byId("cfg-env-vars");
    if (envEl) envEl.value = envLines;
  }

  async function handleConfigSwitch() {
    const select = byId("config-select");
    if (!select || !select.value) return;
    const name = select.value;
    try {
      await post("/config/switch", { config: name });
      addEvent(`Switched config to ${name}`);
      await loadCurrentConfig();
      await loadConfigList();
      await refreshAll();
    } catch (error) {
      addEvent(`Config switch: ${error.message}`);
    }
  }

  async function handleConfigSave() {
    const updates = {};
    const fields = {
      "cfg-input-folder": "input_folder",
      "cfg-output-folder": "output_folder",
      "cfg-database-path": "database_path",
      "cfg-mode": "mode",
      "cfg-eps-realism": "eps_realism",
      "cfg-min-samples": "min_samples",
      "cfg-lora-min-images": "lora_min_images",
      "cfg-train-steps": "kohya_train_steps",
      "cfg-learning-rate": "kohya_learning_rate",
      "cfg-network-dim": "kohya_network_dim",
      "cfg-network-alpha": "kohya_network_alpha",
      "cfg-batch-size": "kohya_batch_size",
      "cfg-num-repeats": "kohya_num_repeats",
      "cfg-base-model": "kohya_base_model",
      "cfg-output-name": "kohya_output_name",
    };
    for (const [id, key] of Object.entries(fields)) {
      const el = byId(id);
      if (!el) continue;
      const val = el.value !== undefined ? el.value : el.textContent;
      const trimmed = String(val || "").trim();
      if (trimmed === "") continue;
      if (key === "kohya_train_steps" || key === "min_samples" || key === "lora_min_images" || key === "kohya_network_dim" || key === "kohya_network_alpha" || key === "kohya_batch_size" || key === "kohya_num_repeats") {
        const num = Number(trimmed);
        if (Number.isFinite(num)) updates[key] = num;
      } else if (key === "eps_realism") {
        const num = Number(trimmed);
        if (Number.isFinite(num)) updates[key] = num;
      } else if (key === "kohya_learning_rate") {
        const num = Number(trimmed);
        if (Number.isFinite(num)) updates[key] = num;
      } else {
        updates[key] = trimmed;
      }
    }
    // Environment variables from textarea
    const envEl = byId("cfg-env-vars");
    if (envEl) {
      const envText = String(envEl.value || "").trim();
      const envObj = {};
      if (envText) {
        for (const line of envText.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith("#")) continue;
          const eqIdx = trimmed.indexOf("=");
          if (eqIdx > 0) {
            const k = trimmed.slice(0, eqIdx).trim();
            const v = trimmed.slice(eqIdx + 1).trim();
            if (k) envObj[k] = v;
          }
        }
      }
      updates.env = envObj;
    }

    try {
      await put("/config", { updates });
      addEvent("Config saved");
      await loadCurrentConfig();
    } catch (error) {
      addEvent(`Config save failed: ${error.message}`);
    }
  }

  function applyScanSummary(result, fromFreshScan) {
    const data = scanPayloadFromResponse(result);
    const discovered = Number(data.discovered_count || 0);
    const processed = Number(data.processed_count || 0);
    const identities = Number(data.identity_count || 0);
    const review = Number(data.noise_count || 0) + Number(data.no_face_count || 0);
    setStatValues({
      faces_found: discovered,
      new_processed: processed,
      identities,
      review_count: review,
    });
    const opener = fromFreshScan ? "Fresh rebuild complete." : "Scan complete.";
    setText(
      "last-action-text",
      `${opener} ${numberText(identities)} character groups currently tracked.`
    );
    addEvent(
      `${opener} Found ${numberText(discovered)} files, processed ${numberText(processed)}, grouped ${numberText(identities)} character groups.`
    );
  }

  async function runScan() {
    const inputFolder = normalizeOptionalPath(byId("input-folder")?.value || "");
    const outputFolder = normalizeOptionalPath(byId("output-folder")?.value || "");
    renderActivity({
      running: true,
      operation: "scan",
      message: "Finding similar faces...",
      elapsed_s: 0,
    });
    const result = await post("/scan", {
      input_folder: inputFolder || null,
      output_folder: outputFolder || null,
    });
    applyScanSummary(result, false);
    await pollActivity();
    await loadIdentities();
    switchView("identities");
  }

  async function runScanRecluster() {
    const inputFolder = normalizeOptionalPath(byId("input-folder")?.value || "");
    const outputFolder = normalizeOptionalPath(byId("output-folder")?.value || "");
    renderActivity({
      running: true,
      operation: "scan_recluster",
      message: "Starting fresh and rebuilding groups...",
      elapsed_s: 0,
    });
    const result = await post("/scan-recluster", {
      input_folder: inputFolder || null,
      output_folder: outputFolder || null,
    });
    applyScanSummary(result, true);
    await pollActivity();
    await loadIdentities();
    switchView("identities");
  }

  async function exportLora() {
    const minImages = parseInt(byId("min-images")?.value || "5", 10);
    const outputFolder = normalizeOptionalPath(byId("output-folder")?.value || "");
    const selectedIds = getSelectedIdsForExport();
    if (selectedIds && !selectedIds.length) {
      setText("training-status", "Select at least one character group first, or turn off 'Use only selected character groups'.");
      setText("training-detail", "");
      addEvent("Export skipped because no character groups are selected.");
      return;
    }
    renderActivity({
      running: true,
      operation: "export_lora",
      message: "Preparing training set...",
      elapsed_s: 0,
    });
    const result = await post("/export-lora", {
      min_images: Number.isFinite(minImages) ? minImages : 5,
      output_folder: outputFolder || null,
      identity_ids: selectedIds && selectedIds.length ? selectedIds : null,
    });
    const data = exportPayloadFromResponse(result);
    const requestedIds = Array.isArray(data.requested_identity_ids) ? data.requested_identity_ids : [];
    const out = String(data.output_folder || "").trim();
    setText(
      "training-status",
      requestedIds.length
        ? `Dataset ready from ${numberText(data.identities_exported)} selected groups (${numberText(data.images_exported)} images).`
        : `Dataset ready: ${numberText(data.identities_exported)} character groups and ${numberText(data.images_exported)} images.`
    );
    setText("training-detail", out ? `Dataset folder: ${out}` : "");
    setText("last-action-text", "Training dataset export completed.");
    addEvent("Training set export completed.");
    await pollActivity();
  }

  async function trainLora() {
    const minImages = parseInt(byId("min-images")?.value || "5", 10);
    const outputFolder = normalizeOptionalPath(byId("output-folder")?.value || "");
    const selectedIds = getSelectedIdsForExport();
    if (selectedIds && !selectedIds.length) {
      setText("training-status", "Select at least one character group first, or turn off 'Use only selected character groups'.");
      setText("training-detail", "");
      addEvent("Training skipped because no character groups are selected.");
      return;
    }
    renderActivity({
      running: true,
      operation: "train_lora",
      message: "Starting trainer...",
      elapsed_s: 0,
    });
    const result = await post("/train-lora", {
      output_folder: outputFolder || null,
      min_images: Number.isFinite(minImages) ? minImages : 5,
      identity_ids: selectedIds && selectedIds.length ? selectedIds : null,
      prepare_dataset: true,
    });
    const data = trainPayloadFromResponse(result);
    const accepted = Boolean(data.accepted);
    const jobId = String(data.job_id || "").trim();
    const jobStatus = String(data.status || "").trim();
    const returnCode = Number(data.returncode);
    const exportInfo = data && typeof data.export_result === "object" ? data.export_result : null;
    const outputDir = String(data.output_dir || "").trim();
    const datasetDir = String(data.dataset_dir || "").trim();
    const logFile = String(data.log_file || "").trim();
    const stderrLine = firstNonEmptyLine(data.stderr);
    const stdoutLine = firstNonEmptyLine(data.stdout);
    const newArtifacts = Array.isArray(data.new_artifacts) ? data.new_artifacts : [];
    const artifactsAfter = Number(data.artifacts_after);

    if (exportInfo && Number.isFinite(Number(exportInfo.images_exported))) {
      addEvent(
        `Prepared ${numberText(exportInfo.identities_exported)} groups and ${numberText(exportInfo.images_exported)} images before training.`
      );
    }
    if (accepted) {
      setText("training-status", "Training started in background.");
      setText(
        "training-detail",
        jobId
          ? `Job ID: ${jobId}${jobStatus ? ` | Status: ${jobStatus}` : ""}`
          : "Background training request accepted."
      );
      addEvent(jobId ? `Training started. Job ID: ${jobId}.` : "Training started in background.");
      setText("last-action-text", "Training started in background.");
      await pollActivity();
      return;
    }
    if (Number.isFinite(returnCode)) {
      if (returnCode === 0) {
        if (newArtifacts.length > 0) {
          setText("training-status", `Trainer finished successfully. New LoRA file: ${basename(newArtifacts[0])}`);
          addEvent(`Training completed. New LoRA file: ${basename(newArtifacts[0])}.`);
        } else if (Number.isFinite(artifactsAfter) && artifactsAfter > 0) {
          setText("training-status", `Trainer finished successfully. Output folder currently has ${numberText(artifactsAfter)} LoRA file(s).`);
          addEvent("Training completed. No new file name was detected this run.");
        } else {
          setText("training-status", "Trainer finished, but no LoRA files were found in the output folder.");
          addEvent("Training finished but produced no LoRA files.");
        }
      } else {
        setText("training-status", `Trainer finished with code ${returnCode}.`);
        addEvent(`Training command returned code ${returnCode}.`);
      }
    } else {
      setText("training-status", "Training command sent.");
      addEvent("Training command sent.");
    }
    if (outputDir && logFile) {
      setText("training-detail", `Output folder: ${outputDir} | Log: ${logFile}`);
    } else if (outputDir) {
      setText("training-detail", `Output folder: ${outputDir}`);
    } else if (logFile) {
      setText("training-detail", `Run log: ${logFile}`);
    } else if (datasetDir) {
      setText("training-detail", `Dataset folder used: ${datasetDir}`);
    } else if (stderrLine || stdoutLine) {
      setText("training-detail", stderrLine || stdoutLine);
    } else {
      setText("training-detail", "");
    }
    if (logFile) {
      addEvent(`Training log saved: ${logFile}`);
    }
    if (stderrLine && returnCode !== 0) {
      addEvent(`Trainer error: ${stderrLine}`);
    }
    setText("last-action-text", "Training action executed.");
    await pollActivity();
  }

  async function pauseTraining() {
    const result = await post("/pause-training", {});
    const data = result && typeof result === "object"
      ? (result.result && typeof result.result === "object" ? result.result : result)
      : {};
    const ok = Boolean(data.ok);
    const message = String(data.message || "").trim();
    const jobId = String(data.job_id || "").trim();
    const status = String(data.status || "").trim();
    if (ok) {
      setText("training-status", "Pause requested. Training will stop after the current step.");
      setText(
        "training-detail",
        jobId
          ? `Job ID: ${jobId}${status ? ` | Status: ${status}` : ""}`
          : (message || "Pause requested.")
      );
      addEvent(jobId ? `Pause requested for job ${jobId}.` : "Pause requested.");
      renderActivity({
        running: true,
        operation: "train_lora",
        stage: "pausing",
        message: message || "Pause requested. Stopping after current step...",
      });
      window.setTimeout(() => {
        void loadPausedJobs();
      }, 2500);
      return;
    }
    setText("training-status", message || "No active training job to pause.");
    if (jobId || status) {
      setText("training-detail", `Job ID: ${jobId || "n/a"}${status ? ` | Status: ${status}` : ""}`);
    }
    addEvent(message || "Pause request was not accepted.");
    await pollActivity();
  }

  async function resumeTraining() {
    const selected = getSelectedPausedJob();
    if (!selected) {
      setText("training-status", "Pick a paused run first.");
      setText("training-detail", "");
      return;
    }
    const sourceJobId = String(selected.job_id || "").trim();
    if (!sourceJobId) {
      setText("training-status", "Selected paused run is missing a job ID.");
      setText("training-detail", "");
      return;
    }
    const response = await post("/resume-training", {
      job_id: sourceJobId,
      prepare_dataset: false,
    });
    const data = trainPayloadFromResponse(response);
    if (Boolean(data.accepted)) {
      const newJobId = String(data.job_id || "").trim();
      setText("training-status", "Resumed training started in background.");
      setText(
        "training-detail",
        `Resumed from ${sourceJobId}${newJobId ? ` | New Job ID: ${newJobId}` : ""}`
      );
      addEvent(
        newJobId
          ? `Resumed training: ${sourceJobId} -> ${newJobId}.`
          : `Resumed training from ${sourceJobId}.`
      );
      setText("last-action-text", "Resumed training in background.");
      await pollActivity();
      await loadPausedJobs(sourceJobId);
      return;
    }
    const message = String(data.message || "Resume request was not accepted.");
    setText("training-status", message);
    setText("training-detail", sourceJobId ? `Source Job ID: ${sourceJobId}` : "");
    addEvent(`Resume request failed: ${message}`);
  }

  function basename(path) {
    const raw = String(path || "");
    const slash = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
    return slash >= 0 ? raw.slice(slash + 1) : raw;
  }

  function updateIdentityDetailButtons() {
    const loadMore = byId("identity-load-more-btn");
    const refreshDetail = byId("identity-refresh-btn");
    const saveLabel = byId("identity-label-save-btn");
    const clearLabel = byId("identity-label-clear-btn");
    const labelInput = byId("identity-label-input");
    const hasSelection = Boolean(selectedIdentityId);
    if (refreshDetail) refreshDetail.disabled = !hasSelection;
    if (saveLabel) saveLabel.disabled = !hasSelection;
    if (clearLabel) clearLabel.disabled = !hasSelection;
    if (labelInput) labelInput.disabled = !hasSelection;
    if (loadMore) {
      const hasMore = hasSelection && selectedIdentityOffset < selectedIdentityTotal;
      loadMore.disabled = !hasMore;
      loadMore.style.display = hasMore ? "inline-flex" : "none";
    }
  }

  function renderIdentityDetail(detail, append) {
    const meta = byId("identity-detail-meta");
    const grid = byId("identity-images-grid");
    const labelInput = byId("identity-label-input");
    const feedback = byId("identity-detail-feedback");
    if (!meta || !grid) return;

    if (!append) {
      grid.innerHTML = "";
    }

    if (!detail || typeof detail !== "object" || !detail.identity_id) {
      meta.textContent = "Select a character row to preview photos.";
      if (labelInput) labelInput.value = "";
      if (feedback) feedback.textContent = "Tip: review each group and name the character you want to keep.";
      updateIdentityDetailButtons();
      return;
    }

    const label = detail.label ? String(detail.label) : "(not named yet)";
    if (labelInput) labelInput.value = detail.label ? String(detail.label) : "";
    const loadedCount = Math.min(selectedIdentityOffset, selectedIdentityTotal);
    meta.textContent =
      `Group ${detail.identity_id} | ${detail.member_count ?? 0} photos | Name: ${label} | Showing ${loadedCount}/${selectedIdentityTotal}`;

    const rows = Array.isArray(detail.images) ? detail.images : [];
    if (!rows.length && !append) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No photos found in this character group page.";
      grid.appendChild(empty);
      if (feedback) feedback.textContent = "No photos loaded yet.";
      updateIdentityDetailButtons();
      return;
    }

    for (const item of rows) {
      const path = String(item.path || "");
      const status = String(item.status || "");
      const exists = Boolean(item.exists);

      const card = document.createElement("div");
      card.className = "identity-image-card";

      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = basename(path);
      img.src = `${API_BASE}/image?${new URLSearchParams({ path }).toString()}`;
      if (!exists) img.style.opacity = "0.35";
      img.addEventListener("error", () => {
        img.style.opacity = "0.35";
      });
      card.appendChild(img);

      const name = document.createElement("div");
      name.className = "identity-image-name";
      name.title = path;
      name.textContent = basename(path);
      card.appendChild(name);

      const state = document.createElement("div");
      state.className = "identity-image-status";
      if (status === "blacklisted") state.textContent = "Hidden from future scans";
      else if (status === "removed") state.textContent = "Removed from this character group";
      else if (status === "assigned") state.textContent = "In this character group";
      else state.textContent = status || "Needs review";
      if (!exists) state.textContent += " (file missing)";
      card.appendChild(state);

      const actions = document.createElement("div");
      actions.className = "identity-image-actions";

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-secondary";
      removeBtn.textContent = "Remove";
      removeBtn.disabled = !exists && status === "removed";
      removeBtn.addEventListener("click", async () => {
        await runImageAction(path, "remove");
      });
      actions.appendChild(removeBtn);

      const blacklistBtn = document.createElement("button");
      blacklistBtn.type = "button";
      blacklistBtn.className = "btn btn-danger";
      blacklistBtn.textContent = "Hide Future Matches";
      blacklistBtn.disabled = status === "blacklisted";
      blacklistBtn.addEventListener("click", async () => {
        await runImageAction(path, "blacklist");
      });
      actions.appendChild(blacklistBtn);

      const restoreBtn = document.createElement("button");
      restoreBtn.type = "button";
      restoreBtn.className = "btn btn-secondary";
      restoreBtn.textContent = "Restore";
      restoreBtn.disabled = !(status === "blacklisted" || status === "removed");
      restoreBtn.addEventListener("click", async () => {
        await runImageAction(path, "restore");
      });
      actions.appendChild(restoreBtn);

      card.appendChild(actions);
      grid.appendChild(card);
    }

    if (feedback) feedback.textContent = `Loaded ${loadedCount} of ${selectedIdentityTotal} photos for this character group.`;
    updateIdentityDetailButtons();
  }

  async function loadIdentityDetail(options) {
    const opts = options && typeof options === "object" ? options : {};
    const append = Boolean(opts.append);
    const feedback = byId("identity-detail-feedback");
    if (!selectedIdentityId) {
      renderIdentityDetail(null, false);
      return;
    }

    const offset = append ? selectedIdentityOffset : 0;
    const query = new URLSearchParams({
      limit: String(IDENTITY_PAGE_SIZE),
      offset: String(offset),
    });

    const detail = await get(`/identity/${selectedIdentityId}?${query.toString()}`);
    selectedIdentityTotal = Number(detail.images_total || 0);
    const rows = Array.isArray(detail.images) ? detail.images : [];
    selectedIdentityRows = append ? selectedIdentityRows.concat(rows) : rows;
    selectedIdentityOffset = append ? selectedIdentityOffset + rows.length : rows.length;

    const mergedDetail = { ...detail, images: selectedIdentityRows };
    renderIdentityDetail(mergedDetail, false);
    if (feedback) feedback.textContent = `Showing ${selectedIdentityOffset} of ${selectedIdentityTotal} photos.`;
  }

  async function runImageAction(imagePath, action) {
    const feedback = byId("identity-detail-feedback");
    const payload = { image_path: String(imagePath || ""), action: String(action || "") };
    try {
      const result = await post("/image/action", payload);
      const data = result && result.result && typeof result.result === "object" ? result.result : result;
      const nextStatus = String(data.after_status || "").trim();
      if (feedback) feedback.textContent = `Saved. This photo is now marked as "${nextStatus || "updated"}".`;
      addEvent(`Image updated in character review (${action}).`);
      await loadIdentities();
      await loadIdentityDetail({ append: false });
    } catch (error) {
      if (feedback) feedback.textContent = `Could not update image: ${error.message}`;
      addEvent(`Image update failed: ${error.message}`);
    }
  }

  async function saveIdentityLabel(clearLabel) {
    const feedback = byId("identity-detail-feedback");
    const labelInput = byId("identity-label-input");
    if (!selectedIdentityId) return;
    const raw = labelInput ? String(labelInput.value || "").trim() : "";
    const label = clearLabel ? null : (raw || null);
    try {
      await post("/label", {
        identity_id: Number(selectedIdentityId),
        label,
      });
      if (feedback) {
        feedback.textContent = label
          ? `Saved character name: ${label}`
          : "Character name cleared for this group.";
      }
      addEvent(label ? `Named group ${selectedIdentityId} as "${label}".` : `Cleared name for group ${selectedIdentityId}.`);
      await loadIdentities();
      await loadIdentityDetail({ append: false });
    } catch (error) {
      if (feedback) feedback.textContent = `Could not save name: ${error.message}`;
      addEvent(`Name update failed: ${error.message}`);
    }
  }

  function setSelectedIdentity(identityId) {
    selectedIdentityId = Number(identityId || 0) || null;
    selectedIdentityOffset = 0;
    selectedIdentityTotal = 0;
    selectedIdentityRows = [];
    renderIdentityRows(identitiesCache || []);
    void loadIdentityDetail({ append: false });
  }

  function renderIdentityRows(identities) {
    const container = byId("identity-table");
    if (!container) return;
    container.innerHTML = "";

    const rows = Array.isArray(identities) ? identities : [];
    identitiesCache = rows;

    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "identity-card-empty";
      empty.textContent = "No character groups found for this filter.";
      container.appendChild(empty);
      updateSelectedExportSummary();
      return;
    }

    for (const item of rows) {
      const identityId = Number(item.identity_id || 0);
      const label = String(item.label || "Unlabeled");
      const memberCount = Number(item.member_count || 0);
      const updatedAt = String(item.updated_at || "");

      const card = document.createElement("div");
      card.className = "identity-row-card";
      if (selectedIdentityId && identityId === selectedIdentityId) {
        card.classList.add("selected");
      }

      // Checkbox
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "identity-row-cb";
      cb.checked = selectedExportIdentityIds.has(identityId);
      cb.addEventListener("click", (event) => {
        event.stopPropagation();
      });
      cb.addEventListener("change", () => {
        if (cb.checked) selectedExportIdentityIds.add(identityId);
        else selectedExportIdentityIds.delete(identityId);
        updateSelectedExportSummary();
      });
      card.appendChild(cb);

      // Main info
      const info = document.createElement("div");
      info.className = "identity-row-info";

      const nameEl = document.createElement("div");
      nameEl.className = "identity-row-name";
      nameEl.textContent = label;
      info.appendChild(nameEl);

      const metaEl = document.createElement("div");
      metaEl.className = "identity-row-meta";
      metaEl.textContent = `Group ${identityId} · ${memberCount} photo${memberCount !== 1 ? "s" : ""}`;
      info.appendChild(metaEl);

      card.appendChild(info);

      // Photo count badge
      const countEl = document.createElement("div");
      countEl.className = "identity-row-count";
      countEl.textContent = String(memberCount);
      card.appendChild(countEl);

      card.addEventListener("click", () => {
        setSelectedIdentity(identityId);
      });
      container.appendChild(card);
    }
    updateSelectedExportSummary();
  }

  async function loadIdentities() {
    const minMembers = parseInt(byId("min-members")?.value || "1", 10);
    const query = `?min_members=${Number.isFinite(minMembers) ? minMembers : 1}`;
    const result = await get(`/identities${query}`);
    const rows = Array.isArray(result.identities) ? result.identities : [];
    renderIdentityRows(rows);

    setStatValues({ identities: rows.length });

    if (selectedIdentityId) {
      const stillPresent = rows.some((row) => Number(row.identity_id) === selectedIdentityId);
      if (!stillPresent) {
        selectedIdentityId = null;
        selectedIdentityOffset = 0;
        selectedIdentityTotal = 0;
        selectedIdentityRows = [];
        renderIdentityDetail(null, false);
      }
    }
    updateIdentityDetailButtons();
    return result;
  }

  async function refreshAll() {
    await loadHealth();
    await loadIdentities();
    await loadPausedJobs();
    setText("last-action-text", "Workspace refreshed.");
  }

  async function init() {
    renderEventFeed();
    setStatValues({
      faces_found: 0,
      new_processed: 0,
      identities: 0,
      review_count: 0,
    });
    updateSelectedExportSummary();

    document.querySelectorAll(".nav-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        switchView(tab.dataset.view || "studio");
      });
    });

    byId("refresh-all-btn")?.addEventListener("click", async () => {
      setButtonBusy("refresh-all-btn", true, "Refreshing...");
      try {
        await refreshAll();
        addEvent("Workspace refreshed.");
      } catch (error) {
        setText("last-action-text", `Refresh failed: ${error.message}`);
        addEvent(`Refresh failed: ${error.message}`);
      } finally {
        setButtonBusy("refresh-all-btn", false, "Refresh");
      }
    });

    byId("scan-btn")?.addEventListener("click", async () => {
      setButtonBusy("scan-btn", true, "Scanning...");
      try {
        await runScan();
      } catch (error) {
        setText("last-action-text", `Scan failed: ${error.message}`);
        addEvent(`Scan failed: ${error.message}`);
        renderActivity({
          running: false,
          operation: "scan",
          message: "Scan failed.",
          last_error: String(error?.message || "scan_failed"),
        });
      } finally {
        setButtonBusy("scan-btn", false, "Find Similar Faces");
      }
    });

    byId("scan-recluster-btn")?.addEventListener("click", async () => {
      const confirmed = window.confirm(
        "Start Fresh will reset DNADuck's saved groups and rebuild character groups from your current folder. Continue?"
      );
      if (!confirmed) return;
      setButtonBusy("scan-recluster-btn", true, "Rebuilding...");
      try {
        await runScanRecluster();
      } catch (error) {
        setText("last-action-text", `Fresh rebuild failed: ${error.message}`);
        addEvent(`Fresh rebuild failed: ${error.message}`);
        renderActivity({
          running: false,
          operation: "scan_recluster",
          message: "Fresh rebuild failed.",
          last_error: String(error?.message || "scan_recluster_failed"),
        });
      } finally {
        setButtonBusy("scan-recluster-btn", false, "Start Fresh (Rebuild)");
      }
    });

    byId("export-btn")?.addEventListener("click", async () => {
      setButtonBusy("export-btn", true, "Preparing...");
      try {
        await exportLora();
      } catch (error) {
        setText("training-status", `Could not prepare dataset: ${error.message}`);
        setText("training-detail", "");
        addEvent(`Dataset export failed: ${error.message}`);
        renderActivity({
          running: false,
          operation: "export_lora",
          message: "Export failed.",
          last_error: String(error?.message || "export_failed"),
        });
      } finally {
        setButtonBusy("export-btn", false, "Prepare Training Set");
      }
    });

    byId("train-btn")?.addEventListener("click", async () => {
      setButtonBusy("train-btn", true, "Starting...");
      try {
        await trainLora();
      } catch (error) {
        setText("training-status", `Training launch failed: ${error.message}`);
        setText("training-detail", "");
        addEvent(`Training launch failed: ${error.message}`);
        renderActivity({
          running: false,
          operation: "train_lora",
          message: "Training launch failed.",
          last_error: String(error?.message || "train_failed"),
        });
      } finally {
        setButtonBusy("train-btn", false, "Start Training");
      }
    });

    byId("pause-train-btn")?.addEventListener("click", async () => {
      setButtonBusy("pause-train-btn", true, "Pausing...");
      try {
        await pauseTraining();
      } catch (error) {
        setText("training-status", `Pause request failed: ${error.message}`);
        addEvent(`Pause request failed: ${error.message}`);
      } finally {
        setButtonBusy("pause-train-btn", false, "Pause Training");
      }
    });

    byId("paused-job-select")?.addEventListener("change", () => {
      updatePausedJobSummary();
    });

    byId("paused-refresh-btn")?.addEventListener("click", async () => {
      setButtonBusy("paused-refresh-btn", true, "Refreshing...");
      try {
        await loadPausedJobs();
        addEvent("Paused run list refreshed.");
      } catch (error) {
        addEvent(`Could not refresh paused runs: ${error.message}`);
      } finally {
        setButtonBusy("paused-refresh-btn", false, "Refresh");
      }
    });

    byId("resume-train-btn")?.addEventListener("click", async () => {
      setButtonBusy("resume-train-btn", true, "Resuming...");
      try {
        await resumeTraining();
      } catch (error) {
        setText("training-status", `Resume failed: ${error.message}`);
        setText("training-detail", "");
        addEvent(`Resume failed: ${error.message}`);
      } finally {
        setButtonBusy("resume-train-btn", false, "Resume Selected");
      }
    });

    byId("refresh-identities-btn")?.addEventListener("click", async () => {
      setButtonBusy("refresh-identities-btn", true, "Refreshing...");
      try {
        await loadIdentities();
        addEvent("Character list refreshed.");
      } catch (error) {
        setText("last-action-text", `Could not refresh character groups: ${error.message}`);
      } finally {
        setButtonBusy("refresh-identities-btn", false, "Refresh Characters");
      }
    });

    byId("select-visible-btn")?.addEventListener("click", () => {
      const rows = Array.isArray(identitiesCache) ? identitiesCache : [];
      for (const row of rows) {
        const identityId = Number(row.identity_id || 0);
        if (identityId > 0) selectedExportIdentityIds.add(identityId);
      }
      renderIdentityRows(identitiesCache);
      addEvent(`Selected ${numberText(selectedExportIdentityIds.size)} character groups.`);
    });

    byId("clear-selected-btn")?.addEventListener("click", () => {
      selectedExportIdentityIds = new Set();
      renderIdentityRows(identitiesCache);
      addEvent("Cleared selected character groups.");
    });

    byId("identity-refresh-btn")?.addEventListener("click", async () => {
      setButtonBusy("identity-refresh-btn", true, "Refreshing...");
      try {
        await loadIdentityDetail({ append: false });
      } finally {
        setButtonBusy("identity-refresh-btn", false, "Refresh Detail");
      }
    });

    byId("identity-load-more-btn")?.addEventListener("click", async () => {
      setButtonBusy("identity-load-more-btn", true, "Loading...");
      try {
        await loadIdentityDetail({ append: true });
      } catch (error) {
        setText("identity-detail-feedback", `Could not load more photos: ${error.message}`);
      } finally {
        setButtonBusy("identity-load-more-btn", false, "Load More");
      }
    });

    byId("identity-label-save-btn")?.addEventListener("click", async () => {
      setButtonBusy("identity-label-save-btn", true, "Saving...");
      try {
        await saveIdentityLabel(false);
      } finally {
        setButtonBusy("identity-label-save-btn", false, "Save Name");
      }
    });

    byId("identity-label-clear-btn")?.addEventListener("click", async () => {
      setButtonBusy("identity-label-clear-btn", true, "Clearing...");
      try {
        await saveIdentityLabel(true);
      } finally {
        setButtonBusy("identity-label-clear-btn", false, "Clear");
      }
    });

    byId("identity-label-input")?.addEventListener("keydown", async (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      setButtonBusy("identity-label-save-btn", true, "Saving...");
      try {
        await saveIdentityLabel(false);
      } finally {
        setButtonBusy("identity-label-save-btn", false, "Save Name");
      }
    });

    byId("min-members")?.addEventListener("change", async () => {
      try {
        await loadIdentities();
      } catch (error) {
        setText("last-action-text", `Could not filter character groups: ${error.message}`);
      }
    });

    byId("use-selected-only")?.addEventListener("change", () => {
      updateSelectedExportSummary();
    });

    // ── Config event handlers ────────────────────────────────────────

    byId("config-select")?.addEventListener("change", () => {
      void handleConfigSwitch();
    });

    byId("config-save-btn")?.addEventListener("click", async () => {
      byId("config-save-btn").disabled = true;
      try {
        await handleConfigSave();
      } catch (error) {
        addEvent(`Config save failed: ${error.message}`);
      } finally {
        byId("config-save-btn").disabled = false;
      }
    });

    byId("cfg-tab-btn")?.addEventListener("click", () => {
      switchView("config");
    });

    try {
      await pollActivityAndReschedule();
      renderIdentityDetail(null, false);
      updateIdentityDetailButtons();
      await loadHealth();
      await loadConfigList();
      await loadCurrentConfig();
      await loadPausedJobs();
      addEvent("DNADuck is ready.");
    } catch (error) {
      setText("last-action-text", `Startup issue: ${error.message}`);
      setConnectionInfo({ _plugin_mode: "error" });
      addEvent(`Startup issue: ${error.message}`);
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
