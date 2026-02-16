(function () {
  const API_BASE = window.location.pathname.replace(/\/ui\/.*$/, "/api");
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

  function formatDuration(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds || 0)));
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins <= 0) return `${secs}s`;
    return `${mins}m ${secs.toString().padStart(2, "0")}s`;
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
    const running = Boolean(active.running ?? payload.running);
    const operation = String(active.operation || payload.operation || "").trim();
    const stage = String(active.stage || payload.stage || "").trim();
    const lastError = String(active.last_error || payload.last_error || "").trim();

    dot.classList.remove("running", "error");
    if (running) {
      dot.classList.add("running");
      if (operation === "scan_recluster") state.textContent = "Starting Fresh Scan";
      else if (operation === "scan") state.textContent = "Finding Similar Faces";
      else if (operation === "export_lora") state.textContent = "Preparing Training Set";
      else if (operation === "train_lora") state.textContent = "Launching Training";
      else state.textContent = "Working";
      if (stage) state.textContent += ` (${stage})`;
    } else if (lastError) {
      dot.classList.add("error");
      state.textContent = "Needs Attention";
    } else {
      state.textContent = "Ready";
    }

    const msg = String(active.message || payload.message || "").trim();
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

    let barWidth = 0;
    if (Number.isFinite(processed) && Number.isFinite(total) && total > 0) {
      barWidth = Math.max(0, Math.min(100, (processed / total) * 100));
      progress.textContent = `Progress: ${numberText(processed)} of ${numberText(total)} images`;
    } else if (Number.isFinite(discovered) && discovered > 0 && Number.isFinite(toProcess) && toProcess >= 0) {
      const checked = Math.max(0, discovered - toProcess);
      barWidth = Math.max(0, Math.min(100, (checked / discovered) * 100));
      progress.textContent = `Checked ${numberText(checked)} of ${numberText(discovered)} files`;
    } else if (running) {
      barWidth = 20;
      progress.textContent = "Preparing files...";
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

    if (!running && lastError) {
      setText("last-action-text", `Could not finish: ${lastError}`);
      addEvent(`Action failed: ${lastError}`);
    } else if (!running && stage === "complete") {
      setText("last-action-text", "Latest action completed successfully.");
    }

    if (Number.isFinite(assigned) || Number.isFinite(noise) || Number.isFinite(noFace)) {
      const grouped = Number.isFinite(assigned) ? assigned : 0;
      const review = (Number.isFinite(noise) ? noise : 0) + (Number.isFinite(noFace) ? noFace : 0);
      setStatValues({ review_count: review, new_processed: Number.isFinite(processed) ? processed : null });
      if (!running) {
        addEvent(`Grouped ${numberText(grouped)} images. ${numberText(review)} need review.`);
      }
    }
  }

  function isActivityRunning(activity) {
    const payload = activity && typeof activity === "object" ? activity : {};
    const remote = payload.remote_activity && typeof payload.remote_activity === "object"
      ? payload.remote_activity
      : null;
    const active = remote || payload;
    return Boolean(active.running ?? payload.running);
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
    setText(
      "training-status",
      requestedIds.length
        ? `Dataset ready from ${numberText(data.identities_exported)} selected groups (${numberText(data.images_exported)} images).`
        : `Dataset ready: ${numberText(data.identities_exported)} character groups and ${numberText(data.images_exported)} images.`
    );
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
    const returnCode = Number(data.returncode);
    const exportInfo = data && typeof data.export_result === "object" ? data.export_result : null;
    if (exportInfo && Number.isFinite(Number(exportInfo.images_exported))) {
      addEvent(
        `Prepared ${numberText(exportInfo.identities_exported)} groups and ${numberText(exportInfo.images_exported)} images before training.`
      );
    }
    if (Number.isFinite(returnCode)) {
      if (returnCode === 0) {
        setText("training-status", "Trainer finished successfully.");
        addEvent("Training command completed successfully.");
      } else {
        setText("training-status", `Trainer finished with code ${returnCode}.`);
        addEvent(`Training command returned code ${returnCode}.`);
      }
    } else {
      setText("training-status", "Training command sent.");
      addEvent("Training command sent.");
    }
    setText("last-action-text", "Training action executed.");
    await pollActivity();
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
    const tbody = byId("identity-table")?.querySelector("tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const rows = Array.isArray(identities) ? identities : [];
    identitiesCache = rows;
    for (const item of rows) {
      const tr = document.createElement("tr");
      tr.className = "identity-row";
      const identityId = Number(item.identity_id || 0);
      if (selectedIdentityId && identityId === selectedIdentityId) {
        tr.classList.add("selected");
      }
      const selectTd = document.createElement("td");
      selectTd.className = "identity-select-cell";
      const selectInput = document.createElement("input");
      selectInput.type = "checkbox";
      selectInput.checked = selectedExportIdentityIds.has(identityId);
      selectInput.addEventListener("click", (event) => {
        event.stopPropagation();
      });
      selectInput.addEventListener("change", () => {
        if (selectInput.checked) selectedExportIdentityIds.add(identityId);
        else selectedExportIdentityIds.delete(identityId);
        updateSelectedExportSummary();
      });
      selectTd.appendChild(selectInput);
      tr.appendChild(selectTd);

      const idTd = document.createElement("td");
      idTd.textContent = String(item.identity_id ?? "");
      tr.appendChild(idTd);

      const memberTd = document.createElement("td");
      memberTd.textContent = String(item.member_count ?? "");
      tr.appendChild(memberTd);

      const labelTd = document.createElement("td");
      labelTd.textContent = String(item.label ?? "Unlabeled");
      tr.appendChild(labelTd);

      const updatedTd = document.createElement("td");
      updatedTd.textContent = String(item.updated_at ?? "");
      tr.appendChild(updatedTd);

      tr.addEventListener("click", () => {
        setSelectedIdentity(identityId);
      });
      tbody.appendChild(tr);
    }

    if (!rows.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = '<td colspan="5">No character groups found for this filter.</td>';
      tbody.appendChild(tr);
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

    try {
      await pollActivityAndReschedule();
      renderIdentityDetail(null, false);
      updateIdentityDetailButtons();
      await refreshAll();
      addEvent("DNADuck is ready.");
    } catch (error) {
      setText("last-action-text", `Startup issue: ${error.message}`);
      setConnectionInfo({ _plugin_mode: "error" });
      addEvent(`Startup issue: ${error.message}`);
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();
