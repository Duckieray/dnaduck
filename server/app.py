from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import yaml

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.service import (
    apply_image_action,
    auto_generate,
    cancel_auto_generate,
    count_unassigned,
    export_lora,
    fetch_unassigned_images,
    get_auto_generate_status,
    get_identities,
    get_identity_detail,
    is_active_training_running,
    merge_identity_groups,
    reanalyze_no_face,
    reassign_image,
    recluster_noise,
    relabel_identity,
    request_active_training_stop,
    scan_images,
    scan_recluster_from_scratch,
    search_by_image,
    trigger_lora_training,
)
from core.utils import load_config


class ScanRequest(BaseModel):
    input_folder: str | None = None
    output_folder: str | None = None


class RelabelRequest(BaseModel):
    label: str | None = None


class MergeRequest(BaseModel):
    target_id: int = Field(..., ge=1)
    source_ids: list[int] = Field(..., min_length=1)


class SearchRequest(BaseModel):
    image_path: str
    top_k: int = Field(default=5, ge=1, le=100)


class LoraExportRequest(BaseModel):
    output_folder: str | None = None
    min_images: int | None = Field(default=None, ge=1)
    identity_ids: list[int] | None = None


class LoraTrainRequest(BaseModel):
    output_folder: str | None = None
    min_images: int | None = Field(default=None, ge=1)
    identity_ids: list[int] | None = None
    prepare_dataset: bool = False
    wait_for_result: bool = False


class ResumeTrainRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    prepare_dataset: bool = False


class ImageActionRequest(BaseModel):
    image_path: str
    action: str = Field(..., description="remove | blacklist | restore")


class ReassignImageRequest(BaseModel):
    image_path: str
    identity_id: int = Field(..., ge=1)


class SwitchConfigRequest(BaseModel):
    config: str = Field(..., description="Config filename relative to config directory")


class UpdateConfigRequest(BaseModel):
    updates: dict = Field(..., description="Key-value pairs to update in the config YAML")


class AutoGenerateRequest(BaseModel):
    identity_id: int = Field(..., ge=1)
    target_count: int = Field(default=50, ge=1, le=10000)
    max_attempts: int = Field(default=500, ge=1, le=100000)
    assign_eps_realism: float | None = Field(default=None, ge=0.01, le=1.0)
    assign_eps_anime: float | None = Field(default=None, ge=0.01, le=1.0)
    target_identity_id: int | None = Field(default=None, ge=1)
    new_character_label: str | None = Field(default=None, min_length=1, max_length=100)


_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: dict = {
    "running": False,
    "operation": None,
    "job_id": None,
    "stage": None,
    "message": "Idle",
    "started_at": None,
    "updated_at": time.time(),
    "last_completed_at": None,
    "last_duration_s": None,
    "discovered_count": None,
    "checked_count": None,
    "total_to_process": None,
    "processed_count": None,
    "total_count": None,
    "embedded_count": None,
    "assigned_count": None,
    "noise_count": None,
    "no_face_count": None,
    "identity_count": None,
    "eta_s": None,
    "progress_pct": None,
    "last_result": None,
    "last_error": None,
}
_TRAIN_JOB_LOCK = threading.Lock()
_TRAIN_JOBS: dict[str, dict] = {}
_ACTIVE_TRAIN_JOB_ID: str | None = None
_MAX_TRAIN_JOBS = 100
_TRAIN_JOBS_STATE_PATH: Path | None = None
_ACTIVE_CONFIG_PATH: Path | None = None


def _normalize_optional_path(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw or None


def _new_train_job(payload: dict, *, resumed_from_job_id: str | None = None) -> dict:
    now = time.time()
    return {
        "job_id": uuid.uuid4().hex,
        "operation": "train_lora",
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "request": payload,
        "result": None,
        "error": None,
        "cancel_requested": False,
        "resumed_from_job_id": resumed_from_job_id,
    }


def _state_path_for_config(cfg_path: Path) -> Path:
    cfg = cfg_path.resolve()
    return cfg.parent / ".dnaduck_train_jobs.json"


def _persist_train_jobs_locked() -> None:
    state_path = _TRAIN_JOBS_STATE_PATH
    if state_path is None:
        return
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "jobs": list(_TRAIN_JOBS.values()),
        }
        tmp = state_path.with_name(f"{state_path.name}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(state_path)
    except Exception:
        # Persistence is best-effort; API runtime should continue even on disk errors.
        pass


def _normalize_loaded_train_job(raw: object) -> tuple[dict | None, bool]:
    if not isinstance(raw, dict):
        return None, False
    job_id = str(raw.get("job_id") or "").strip()
    if not job_id:
        return None, False
    now = time.time()

    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    status = str(raw.get("status") or "failed").strip().lower()
    request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
    result = raw.get("result") if isinstance(raw.get("result"), dict) else None
    error = None if raw.get("error") is None else str(raw.get("error"))
    created_at = _safe_float(raw.get("created_at"), now)
    updated_at = _safe_float(raw.get("updated_at"), created_at)
    started_at = raw.get("started_at")
    started_at = _safe_float(started_at, created_at) if isinstance(started_at, (int, float, str)) else None
    finished_at = raw.get("finished_at")
    finished_at = _safe_float(finished_at, updated_at) if isinstance(finished_at, (int, float, str)) else None
    resumed_from_job_id = str(raw.get("resumed_from_job_id") or "").strip() or None
    changed = False

    if status in {"queued", "running", "pausing"}:
        # After process restart, in-flight jobs can no longer be active.
        status = "paused"
        if not error:
            error = "Recovered after API restart."
        if finished_at is None:
            finished_at = now
        changed = True

    if status not in {"queued", "running", "pausing", "paused", "completed", "failed"}:
        status = "failed"
        changed = True
    if finished_at is None and status in {"paused", "completed", "failed"}:
        finished_at = max(updated_at, now)
        changed = True

    job = {
        "job_id": job_id,
        "operation": "train_lora",
        "status": status,
        "created_at": created_at,
        "updated_at": max(updated_at, created_at),
        "started_at": started_at,
        "finished_at": finished_at,
        "request": request,
        "result": result,
        "error": error,
        "cancel_requested": False,
        "resumed_from_job_id": resumed_from_job_id,
    }
    return job, changed


def _load_train_jobs_state(cfg_path: Path) -> None:
    global _TRAIN_JOBS_STATE_PATH, _ACTIVE_TRAIN_JOB_ID
    state_path = _state_path_for_config(cfg_path)
    loaded: dict[str, dict] = {}
    changed = False

    source_count = 0
    if state_path.exists():
        try:
            raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            raw_payload = {}
        rows = raw_payload.get("jobs") if isinstance(raw_payload, dict) else []
        if isinstance(rows, list):
            source_count = len(rows)
            for row in rows:
                job, row_changed = _normalize_loaded_train_job(row)
                if not job:
                    changed = True
                    continue
                loaded[str(job["job_id"])] = job
                changed = changed or row_changed

    with _TRAIN_JOB_LOCK:
        _TRAIN_JOBS_STATE_PATH = state_path
        _TRAIN_JOBS.clear()
        _TRAIN_JOBS.update(loaded)
        _ACTIVE_TRAIN_JOB_ID = None
        _trim_train_jobs()
        if len(_TRAIN_JOBS) != source_count:
            changed = True
        if changed:
            _persist_train_jobs_locked()


def _trim_train_jobs() -> None:
    if len(_TRAIN_JOBS) <= _MAX_TRAIN_JOBS:
        return
    completed = [
        (job.get("finished_at") or 0.0, job_id)
        for job_id, job in _TRAIN_JOBS.items()
        if str(job.get("status", "")).lower() in {"completed", "failed", "paused"}
    ]
    completed.sort(key=lambda row: row[0])
    for _, job_id in completed:
        if len(_TRAIN_JOBS) <= _MAX_TRAIN_JOBS:
            break
        _TRAIN_JOBS.pop(job_id, None)


def _create_train_job(payload: dict, *, resumed_from_job_id: str | None = None) -> dict:
    global _ACTIVE_TRAIN_JOB_ID
    with _TRAIN_JOB_LOCK:
        if _ACTIVE_TRAIN_JOB_ID:
            active = _TRAIN_JOBS.get(_ACTIVE_TRAIN_JOB_ID)
            if active and str(active.get("status", "")).lower() in {"queued", "running", "pausing"}:
                raise RuntimeError(f"A training job is already active: {_ACTIVE_TRAIN_JOB_ID}")
            _ACTIVE_TRAIN_JOB_ID = None
        job = _new_train_job(payload, resumed_from_job_id=resumed_from_job_id)
        _TRAIN_JOBS[job["job_id"]] = job
        _ACTIVE_TRAIN_JOB_ID = job["job_id"]
        _trim_train_jobs()
        _persist_train_jobs_locked()
        return dict(job)


def _normalize_status_filter(status: str | None) -> set[str] | None:
    raw = str(status or "").strip().lower()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _safe_int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    items: list[int] = []
    for raw in value:
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            items.append(parsed)
    return items


def _summarize_train_job(job: dict) -> dict:
    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    identity_ids = _safe_int_list(request.get("identity_ids"))
    new_artifacts = result.get("new_artifacts") if isinstance(result.get("new_artifacts"), list) else []
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "operation": job.get("operation"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "resumed_from_job_id": job.get("resumed_from_job_id"),
        "request_summary": {
            "output_folder": request.get("output_folder"),
            "min_images": request.get("min_images"),
            "identity_ids": identity_ids,
            "identity_count": len(identity_ids),
            "prepare_dataset": bool(request.get("prepare_dataset")),
        },
        "result_summary": {
            "returncode": result.get("returncode"),
            "stopped_by_user": bool(result.get("stopped_by_user")),
            "dataset_dir": result.get("dataset_dir"),
            "output_dir": result.get("output_dir"),
            "log_file": result.get("log_file"),
            "new_artifacts_count": len(new_artifacts),
        },
        "error": job.get("error"),
    }


def _list_train_jobs(*, statuses: set[str] | None = None, limit: int = 25) -> list[dict]:
    with _TRAIN_JOB_LOCK:
        jobs = [dict(row) for row in _TRAIN_JOBS.values()]
    if statuses:
        jobs = [row for row in jobs if str(row.get("status", "")).lower() in statuses]
    jobs.sort(key=lambda row: float(row.get("created_at") or 0.0), reverse=True)
    return [_summarize_train_job(row) for row in jobs[: max(1, int(limit))]]


def _get_train_job(job_id: str) -> dict | None:
    with _TRAIN_JOB_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        return None if job is None else dict(job)


def _get_active_train_job() -> dict | None:
    with _TRAIN_JOB_LOCK:
        if not _ACTIVE_TRAIN_JOB_ID:
            return None
        job = _TRAIN_JOBS.get(_ACTIVE_TRAIN_JOB_ID)
        return None if job is None else dict(job)


def _update_train_job(
    job_id: str,
    *,
    status: str | None = None,
    result: dict | None = None,
    error: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    global _ACTIVE_TRAIN_JOB_ID
    now = time.time()
    with _TRAIN_JOB_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if not job:
            return
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if started and job.get("started_at") is None:
            job["started_at"] = now
        if finished:
            job["finished_at"] = now
            if _ACTIVE_TRAIN_JOB_ID == job_id:
                _ACTIVE_TRAIN_JOB_ID = None
        job["updated_at"] = now
        _persist_train_jobs_locked()


def _request_cancel_train_job(job_id: str) -> bool:
    now = time.time()
    with _TRAIN_JOB_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if not job:
            return False
        status = str(job.get("status", "")).lower()
        if status not in {"queued", "running", "pausing"}:
            return False
        job["cancel_requested"] = True
        if status != "pausing":
            job["status"] = "pausing"
        job["updated_at"] = now
        _persist_train_jobs_locked()
        return True


def _find_cancellable_train_job_id() -> str | None:
    with _TRAIN_JOB_LOCK:
        if _ACTIVE_TRAIN_JOB_ID:
            active = _TRAIN_JOBS.get(_ACTIVE_TRAIN_JOB_ID)
            if active and str(active.get("status", "")).lower() in {"queued", "running", "pausing"}:
                return str(active.get("job_id") or "").strip() or None

        candidates: list[tuple[float, str]] = []
        for job_id, job in _TRAIN_JOBS.items():
            status = str(job.get("status", "")).lower()
            if status in {"queued", "running", "pausing"}:
                updated = float(job.get("updated_at") or job.get("created_at") or 0.0)
                candidates.append((updated, str(job_id)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]


def _is_cancel_requested(job_id: str) -> bool:
    with _TRAIN_JOB_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


def _activity_start(operation: str, message: str, *, job_id: str | None = None) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        for key in (
            "discovered_count",
            "checked_count",
            "total_to_process",
            "processed_count",
            "total_count",
            "embedded_count",
            "assigned_count",
            "noise_count",
            "no_face_count",
            "identity_count",
            "pending_count",
            "eta_s",
            "progress_pct",
        ):
            _ACTIVITY[key] = None
        _ACTIVITY["running"] = True
        _ACTIVITY["operation"] = operation
        _ACTIVITY["job_id"] = job_id
        _ACTIVITY["stage"] = "starting"
        _ACTIVITY["message"] = message
        _ACTIVITY["started_at"] = now
        _ACTIVITY["updated_at"] = now
        _ACTIVITY["last_error"] = None
        _ACTIVITY["last_result"] = None


def _activity_update(payload: dict) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        for key in (
            "stage",
            "message",
            "discovered_count",
            "checked_count",
            "total_to_process",
            "processed_count",
            "total_count",
            "embedded_count",
            "assigned_count",
            "noise_count",
            "no_face_count",
            "identity_count",
            "pending_count",
            "eta_s",
            "progress_pct",
        ):
            if key in payload:
                _ACTIVITY[key] = payload.get(key)
        _ACTIVITY["updated_at"] = now


def _activity_finish(result: dict, message: str) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        started_at = _ACTIVITY.get("started_at")
        duration = None
        if isinstance(started_at, (int, float)):
            duration = max(0.0, now - float(started_at))
        _ACTIVITY["running"] = False
        _ACTIVITY["stage"] = "complete"
        _ACTIVITY["message"] = message
        _ACTIVITY["updated_at"] = now
        _ACTIVITY["last_completed_at"] = now
        _ACTIVITY["last_duration_s"] = duration
        _ACTIVITY["last_result"] = result
        _ACTIVITY["last_error"] = None


def _activity_fail(error_text: str) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        _ACTIVITY["running"] = False
        _ACTIVITY["stage"] = "failed"
        _ACTIVITY["message"] = "Operation failed."
        _ACTIVITY["updated_at"] = now
        _ACTIVITY["last_completed_at"] = now
        _ACTIVITY["last_error"] = str(error_text)[:4000]


def _activity_snapshot() -> dict:
    now = time.time()
    with _ACTIVITY_LOCK:
        payload = dict(_ACTIVITY)
    started_at = payload.get("started_at")
    payload["elapsed_s"] = (
        max(0.0, now - float(started_at))
        if payload.get("running") and isinstance(started_at, (int, float))
        else None
    )
    return payload


def _extract_training_failure_reason(result: dict) -> str | None:
    if not isinstance(result, dict):
        return None
    candidates: list[str] = []
    for key in ("stderr", "stdout"):
        raw = result.get(key)
        if not isinstance(raw, str):
            continue
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        for line in reversed(lines):
            text = line[:300]
            if any(
                token in text
                for token in (
                    "RuntimeError:",
                    "ModuleNotFoundError:",
                    "FileNotFoundError:",
                    "ValueError:",
                    "KeyError:",
                    "CalledProcessError:",
                    "ImportError:",
                )
            ):
                return text
            if "returned non-zero exit status" in text:
                candidates.append(text)
    if candidates:
        return candidates[0]
    return None


def _finalize_paused_if_idle(job_id: str | None) -> bool:
    if not job_id:
        return False
    latest = _get_train_job(str(job_id).strip())
    if not latest:
        return False
    status = str(latest.get("status", "")).lower()
    cancel_requested = bool(latest.get("cancel_requested"))
    if status not in {"queued", "running", "pausing"}:
        return status == "paused"
    if not (cancel_requested or status == "pausing"):
        return False
    if is_active_training_running():
        return False

    result = latest.get("result") if isinstance(latest.get("result"), dict) else {}
    paused_result = dict(result) if isinstance(result, dict) else {}
    paused_result["stopped_by_user"] = True
    paused_result.setdefault("returncode", None)
    _activity_finish(result=paused_result, message="Training paused. Resume to continue.")
    _update_train_job(job_id, status="paused", result=paused_result, finished=True)
    return True


def _execute_train_request(cfg_path: Path, payload: dict, *, stop_callback=None) -> dict:
    if callable(stop_callback):
        try:
            if bool(stop_callback()):
                return {"stopped_by_user": True}
        except Exception:
            pass

    export_result: dict | None = None
    if (
        bool(payload.get("prepare_dataset"))
        or payload.get("min_images") is not None
        or payload.get("identity_ids")
    ):
        _activity_update({"stage": "exporting", "message": "Preparing training set..."})
        export_overrides = {"output_folder": _normalize_optional_path(payload.get("output_folder"))}
        if payload.get("min_images") is not None:
            export_overrides["lora_min_images"] = int(payload["min_images"])
        if payload.get("identity_ids"):
            export_overrides["lora_identity_ids"] = [int(v) for v in payload["identity_ids"] if int(v) > 0]
        export_result = export_lora(config_path=cfg_path, overrides=export_overrides)

    if callable(stop_callback):
        try:
            if bool(stop_callback()):
                result = {"stopped_by_user": True}
                if export_result is not None:
                    result["prepared_dataset"] = True
                    result["export_result"] = export_result
                else:
                    result["prepared_dataset"] = False
                return result
        except Exception:
            pass

    activity_payload: dict[str, object] = {"stage": "training", "message": "Training in progress..."}
    try:
        cfg = load_config(cfg_path.resolve())
        expected_steps = int(cfg.get("kohya_train_steps", 0))
        if expected_steps > 0:
            activity_payload["processed_count"] = 0
            activity_payload["total_count"] = expected_steps
            activity_payload["progress_pct"] = 0.0
            activity_payload["message"] = f"Training in progress... Step 0/{expected_steps}"
    except Exception:
        pass
    _activity_update(activity_payload)
    result = trigger_lora_training(
        config_path=cfg_path,
        overrides={"output_folder": _normalize_optional_path(payload.get("output_folder"))},
        progress_callback=_activity_update,
        stop_callback=stop_callback,
    )
    if export_result is not None:
        result["prepared_dataset"] = True
        result["export_result"] = export_result
    else:
        result["prepared_dataset"] = False
    return result


def _run_train_job(job_id: str, cfg_path: Path, payload: dict) -> None:
    _update_train_job(job_id, status="running", started=True)
    _activity_start(
        operation="train_lora",
        message="Preparing and starting training...",
        job_id=job_id,
    )
    if _is_cancel_requested(job_id):
        result = {"stopped_by_user": True, "returncode": None}
        _activity_finish(result=result, message="Training paused. Resume to continue.")
        _update_train_job(job_id, status="paused", result=result, finished=True)
        return
    try:
        result = _execute_train_request(
            cfg_path=cfg_path,
            payload=payload,
            stop_callback=lambda: _is_cancel_requested(job_id),
        )
        latest = _get_train_job(job_id)
        latest_status = str((latest or {}).get("status", "")).lower()
        if latest_status == "paused":
            # Pause endpoint may have already finalized this job after process stop.
            return

        cancel_requested = _is_cancel_requested(job_id)
        if bool(result.get("stopped_by_user")) or cancel_requested:
            paused_result = dict(result) if isinstance(result, dict) else {}
            paused_result["stopped_by_user"] = True
            paused_result.setdefault("returncode", result.get("returncode") if isinstance(result, dict) else None)
            _activity_finish(result=result, message="Training paused. Resume to continue.")
            _update_train_job(job_id, status="paused", result=paused_result, finished=True)
            return
        returncode = result.get("returncode")
        returncode_int = (
            int(returncode)
            if isinstance(returncode, (int, float, str)) and str(returncode).strip()
            else None
        )
        if returncode_int == 0:
            _activity_finish(result=result, message="Training finished successfully.")
            _update_train_job(job_id, status="completed", result=result, finished=True)
        else:
            reason = _extract_training_failure_reason(result)
            message = (
                f"Training finished with code {returncode_int}."
                if returncode_int is not None
                else "Training command finished."
            )
            if reason:
                message = f"{message} {reason}"
            _activity_finish(result=result, message=message)
            _update_train_job(job_id, status="failed", result=result, finished=True)
    except Exception as exc:
        latest = _get_train_job(job_id)
        if str((latest or {}).get("status", "")).lower() == "paused":
            return
        if _is_cancel_requested(job_id):
            paused_result = {"stopped_by_user": True, "returncode": None}
            _activity_finish(result=paused_result, message="Training paused. Resume to continue.")
            _update_train_job(job_id, status="paused", result=paused_result, finished=True)
            return
        _activity_fail(str(exc))
        _update_train_job(job_id, status="failed", error=str(exc), finished=True)


def create_app(config_path: str | None = None) -> FastAPI:
    from core.utils import configure_logging
    configure_logging("INFO")  # ensure root logger is setup before any autogen/etc logging

    app = FastAPI(title="DNADuck API", version="0.1.0")
    cfg_path = Path(config_path or os.environ.get("DNADUCK_CONFIG", "config.yaml")).resolve()
    global _ACTIVE_CONFIG_PATH
    _ACTIVE_CONFIG_PATH = cfg_path
    _load_train_jobs_state(_ACTIVE_CONFIG_PATH)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "config_path": str(_ACTIVE_CONFIG_PATH)}

    @app.get("/activity")
    def activity() -> dict:
        active = _get_active_train_job()
        if active:
            _finalize_paused_if_idle(str(active.get("job_id") or "").strip())
        return _activity_snapshot()

    @app.get("/jobs/train/active")
    def active_train_job() -> dict:
        job = _get_active_train_job()
        return {"job": job}

    @app.get("/jobs/train")
    def list_train_jobs(
        status: str | None = Query(default=None),
        limit: int = Query(default=25, ge=1, le=200),
    ) -> dict:
        statuses = _normalize_status_filter(status)
        return {"jobs": _list_train_jobs(statuses=statuses, limit=limit)}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = _get_train_job(str(job_id).strip())
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job

    @app.post("/jobs/train/resume")
    def resume_train_job(payload: ResumeTrainRequest) -> dict:
        source_job_id = str(payload.job_id).strip()
        source = _get_train_job(source_job_id)
        if not source:
            raise HTTPException(status_code=404, detail=f"Job not found: {source_job_id}")
        source_status = str(source.get("status", "")).lower()
        if source_status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Only paused jobs can be resumed. Current status: {source_status or 'unknown'}",
            )

        request = source.get("request") if isinstance(source.get("request"), dict) else {}
        resume_payload = {
            "output_folder": _normalize_optional_path(request.get("output_folder")),
            "min_images": None if request.get("min_images") is None else int(request["min_images"]),
            "identity_ids": _safe_int_list(request.get("identity_ids")) or None,
            "prepare_dataset": bool(payload.prepare_dataset),
            "wait_for_result": False,
        }
        try:
            job = _create_train_job(resume_payload, resumed_from_job_id=source_job_id)
            job_id = str(job["job_id"])
            thread = threading.Thread(
                target=_run_train_job,
                kwargs={"job_id": job_id, "cfg_path": _ACTIVE_CONFIG_PATH, "payload": resume_payload},
                daemon=True,
                name=f"dnaduck-train-{job_id[:8]}",
            )
            thread.start()
            return {
                "accepted": True,
                "job_id": job_id,
                "status": "queued",
                "resumed_from_job_id": source_job_id,
                "message": "Training resume started in background.",
            }
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/jobs/train/pause")
    def pause_active_train_job() -> dict:
        job_id = _find_cancellable_train_job_id()
        requested = False
        latest = None
        if job_id:
            requested = _request_cancel_train_job(job_id)
            latest = _get_train_job(job_id)

        stopped_process = request_active_training_stop()
        if requested or stopped_process:
            _activity_update({"stage": "pausing", "message": "Pause requested. Stopping after current step..."})
            finalized = _finalize_paused_if_idle(job_id)
            status = "paused" if finalized else (None if latest is None else latest.get("status"))
            if status is None and stopped_process and not finalized:
                status = "pausing"
            return {
                "ok": True,
                "job_id": job_id,
                "status": status,
                "message": "Pause requested.",
            }

        if not job_id:
            return {"ok": False, "message": "No active training job.", "job_id": None}
        return {
            "ok": False,
            "job_id": job_id,
            "status": None if latest is None else latest.get("status"),
            "message": "Training job is not running.",
        }

    @app.post("/scan")
    def scan(payload: ScanRequest) -> dict:
        overrides = {
            "input_folder": _normalize_optional_path(payload.input_folder),
            "output_folder": _normalize_optional_path(payload.output_folder),
        }
        _activity_start(
            operation="scan",
            message=(
                "Scan started. First run may download InsightFace models; "
                "network and RAM usage can spike."
            ),
        )
        try:
            result = scan_images(
                config_path=_ACTIVE_CONFIG_PATH,
                overrides=overrides,
                progress_callback=_activity_update,
            )
            _activity_finish(result=result, message="Scan complete.")
            return result
        except FileNotFoundError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=500, detail=f"Scan failed: {exc}") from exc

    @app.post("/scan/recluster")
    def scan_recluster(payload: ScanRequest) -> dict:
        overrides = {
            "input_folder": _normalize_optional_path(payload.input_folder),
            "output_folder": _normalize_optional_path(payload.output_folder),
        }
        _activity_start(
            operation="scan_recluster",
            message=(
                "Recluster-from-scratch started. Existing DB identities are reset before scanning."
            ),
        )
        try:
            result = scan_recluster_from_scratch(
                config_path=_ACTIVE_CONFIG_PATH,
                overrides=overrides,
                progress_callback=_activity_update,
            )
            _activity_finish(result=result, message="Recluster-from-scratch complete.")
            return result
        except FileNotFoundError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=500, detail=f"Recluster failed: {exc}") from exc

    @app.get("/identities")
    def identities(min_members: int = 1) -> list[dict]:
        return get_identities(config_path=_ACTIVE_CONFIG_PATH, min_members=min_members)

    @app.get("/identity/{identity_id}")
    def identity_detail(
        identity_id: int,
        limit: int = Query(default=120, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        data = get_identity_detail(
            config_path=_ACTIVE_CONFIG_PATH,
            identity_id=identity_id,
            limit=limit,
            offset=offset,
        )
        if not data:
            raise HTTPException(status_code=404, detail="Identity not found")
        return data

    @app.post("/identity/{identity_id}/label")
    def identity_label(identity_id: int, payload: RelabelRequest) -> dict:
        updated = relabel_identity(config_path=_ACTIVE_CONFIG_PATH, identity_id=identity_id, label=payload.label)
        if not updated:
            raise HTTPException(status_code=404, detail="Identity not found")
        return {"updated": True, "identity_id": identity_id, "label": payload.label}

    @app.post("/identity/merge")
    def identity_merge(payload: MergeRequest) -> dict:
        merge_identity_groups(
            config_path=_ACTIVE_CONFIG_PATH,
            target_id=payload.target_id,
            source_ids=payload.source_ids,
        )
        return {"merged": True, "target_id": payload.target_id, "source_ids": payload.source_ids}

    @app.post("/search")
    def search(payload: SearchRequest) -> dict:
        try:
            rows = search_by_image(
                config_path=_ACTIVE_CONFIG_PATH,
                image_path=Path(_normalize_optional_path(payload.image_path) or ""),
                top_k=payload.top_k,
            )
            return {"matches": rows}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    @app.post("/export/lora")
    def export_lora_endpoint(payload: LoraExportRequest) -> dict:
        _activity_start(operation="export_lora", message="Preparing training set...")
        overrides = {"output_folder": _normalize_optional_path(payload.output_folder)}
        if payload.min_images is not None:
            overrides["lora_min_images"] = int(payload.min_images)
        if payload.identity_ids:
            overrides["lora_identity_ids"] = [int(v) for v in payload.identity_ids if int(v) > 0]
        try:
            result = export_lora(config_path=_ACTIVE_CONFIG_PATH, overrides=overrides)
            _activity_finish(result=result, message="Training set prepared.")
            return result
        except FileNotFoundError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=500, detail=f"LoRA export failed: {exc}") from exc

    @app.post("/train/lora")
    def train_lora_endpoint(payload: LoraTrainRequest) -> dict:
        request_payload = {
            "output_folder": _normalize_optional_path(payload.output_folder),
            "min_images": None if payload.min_images is None else int(payload.min_images),
            "identity_ids": (
                [int(v) for v in payload.identity_ids if int(v) > 0]
                if payload.identity_ids
                else None
            ),
            "prepare_dataset": bool(payload.prepare_dataset),
            "wait_for_result": bool(payload.wait_for_result),
        }
        try:
            job = _create_train_job(request_payload)
            job_id = str(job["job_id"])

            if bool(payload.wait_for_result):
                _run_train_job(job_id=job_id, cfg_path=_ACTIVE_CONFIG_PATH, payload=request_payload)
                completed = _get_train_job(job_id)
                if not completed:
                    raise HTTPException(status_code=500, detail="Training job disappeared unexpectedly.")
                result = completed.get("result")
                if not isinstance(result, dict):
                    detail = str(completed.get("error") or "Training job failed.")
                    raise HTTPException(status_code=500, detail=detail)
                result["job_id"] = job_id
                result["job_status"] = str(completed.get("status") or "")
                return result

            thread = threading.Thread(
                target=_run_train_job,
                kwargs={"job_id": job_id, "cfg_path": _ACTIVE_CONFIG_PATH, "payload": request_payload},
                daemon=True,
                name=f"dnaduck-train-{job_id[:8]}",
            )
            thread.start()
            return {
                "accepted": True,
                "job_id": job_id,
                "status": "queued",
                "message": "Training started in background.",
            }
        except FileNotFoundError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            _activity_fail(str(exc))
            raise HTTPException(status_code=500, detail=f"LoRA training failed: {exc}") from exc

    @app.post("/image/action")
    def image_action(payload: ImageActionRequest) -> dict:
        image_path = _normalize_optional_path(payload.image_path)
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        try:
            return apply_image_action(
                config_path=_ACTIVE_CONFIG_PATH,
                image_path=Path(image_path),
                action=payload.action,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image action failed: {exc}") from exc

    @app.post("/image/reassign")
    def image_reassign(payload: ReassignImageRequest) -> dict:
        image_path = _normalize_optional_path(payload.image_path)
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        try:
            return reassign_image(
                config_path=_ACTIVE_CONFIG_PATH,
                image_path=Path(image_path),
                identity_id=payload.identity_id,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image reassign failed: {exc}") from exc

    @app.get("/images/unassigned")
    def images_unassigned(
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        try:
            return fetch_unassigned_images(
                config_path=_ACTIVE_CONFIG_PATH,
                statuses=("noise", "no_face"),
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to list unassigned images: {exc}") from exc

    @app.get("/images/unassigned/count")
    def images_unassigned_count() -> dict:
        try:
            return count_unassigned(config_path=_ACTIVE_CONFIG_PATH)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to count unassigned images: {exc}") from exc

    @app.post("/images/unassigned/recluster")
    def images_unassigned_recluster() -> dict:
        try:
            return recluster_noise(config_path=_ACTIVE_CONFIG_PATH)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Re-cluster failed: {exc}") from exc

    @app.post("/images/unassigned/reanalyze")
    def images_unassigned_reanalyze() -> dict:
        try:
            return reanalyze_no_face(config_path=_ACTIVE_CONFIG_PATH)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Re-analysis failed: {exc}") from exc

    @app.post("/autogen/start")
    def autogen_start(payload: AutoGenerateRequest) -> dict:
        logging.getLogger("dnaduck.server").warning(
            "autogen_start payload: identity_id=%s target_identity_id=%s new_character_label=%s",
            payload.identity_id, payload.target_identity_id, payload.new_character_label,
        )
        try:
            return auto_generate(
                config_path=_ACTIVE_CONFIG_PATH,
                identity_id=payload.identity_id,
                target_count=payload.target_count,
                max_attempts=payload.max_attempts,
                assign_eps_realism=payload.assign_eps_realism,
                assign_eps_anime=payload.assign_eps_anime,
                target_identity_id=payload.target_identity_id,
                new_character_label=payload.new_character_label,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Auto-generation failed: {exc}") from exc

    @app.post("/autogen/cancel")
    def autogen_cancel() -> dict:
        try:
            return cancel_auto_generate()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cancel failed: {exc}") from exc

    @app.get("/autogen/status")
    def autogen_status() -> dict:
        try:
            return get_auto_generate_status()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Status check failed: {exc}") from exc

    @app.get("/autogen/debug")
    def autogen_debug() -> dict:
        """Dump what autogen sees when it reads the config."""
        try:
            cfg = load_config(_ACTIVE_CONFIG_PATH.resolve())
            ag = cfg.get("auto_generate", {})
            return {
                "config_path": str(_ACTIVE_CONFIG_PATH),
                "auto_generate": {
                    "webbduck_url": ag.get("webbduck_url"),
                    "webbduck_output_dir": ag.get("webbduck_output_dir"),
                    "base_model": ag.get("base_model"),
                    "basic_prompt": ag.get("basic_prompt"),
                    "loras": ag.get("loras"),
                    "embeddings": ag.get("embeddings"),
                    "lora_name": ag.get("lora_name"),
                    "lora_weight": ag.get("lora_weight"),
                    "steps": ag.get("steps"),
                    "cfg": ag.get("cfg"),
                    "scheduler": ag.get("scheduler"),
                    "negative_prompt": ag.get("negative_prompt"),
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/image")
    def image(path: str = Query(...)) -> FileResponse:
        normalized = _normalize_optional_path(path)
        if not normalized:
            raise HTTPException(status_code=400, detail="path is required")
        image_path = Path(normalized).resolve()
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        return FileResponse(str(image_path))

    # ── Config management ────────────────────────────────────────────

    @app.get("/config")
    def get_config() -> dict:
        cfg = load_config(_ACTIVE_CONFIG_PATH.resolve())
        return dict(cfg)

    @app.get("/configs")
    def list_configs() -> dict:
        root = _ACTIVE_CONFIG_PATH.parent.resolve()
        files = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
        current = _ACTIVE_CONFIG_PATH.resolve()
        return {
            "configs": [_safe_rel_path(f, root) for f in files],
            "current": _safe_rel_path(current, root),
            "current_path": str(current),
            "root": str(root),
        }

    def _safe_rel_path(path: Path, anchor: Path) -> str:
        try:
            return str(path.relative_to(anchor))
        except ValueError:
            return str(path)

    @app.post("/config/switch")
    def switch_config(payload: SwitchConfigRequest) -> dict:
        global _ACTIVE_CONFIG_PATH
        root = _ACTIVE_CONFIG_PATH.parent.resolve()
        new_path = (root / payload.config).resolve()
        try:
            new_path = new_path.resolve()
        except Exception:
            pass
        if not new_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {payload.config}")
        _ACTIVE_CONFIG_PATH = new_path
        _load_train_jobs_state(_ACTIVE_CONFIG_PATH)
        return {"ok": True, "config": str(new_path)}

    @app.put("/config")
    def update_config(payload: UpdateConfigRequest) -> dict:
        cfg_path = _ACTIVE_CONFIG_PATH.resolve()
        cfg = load_config(cfg_path)
        for key, value in payload.updates.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return {"ok": True, "updated": list(payload.updates.keys())}

    # ── Static UI files (standalone mode) ────────────────────────────

    _ui_path = Path(__file__).resolve().parent.parent / "integrations" / "webbduck_plugin" / "webapps" / "dnaduck" / "ui"
    if _ui_path.exists():
        app.mount("/ui", StaticFiles(directory=str(_ui_path), html=True), name="ui")

    return app


app = create_app()
