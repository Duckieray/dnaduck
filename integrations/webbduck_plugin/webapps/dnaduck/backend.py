"""DNADuck web plugin backend for WebbDuck."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import yaml

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

PLUGIN_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
STATE_FILE = STATE_DIR / "dnaduck_connection.json"
VALID_CONNECTION_MODES = {"auto", "local_cli", "remote_api", "managed_api"}
REMOTE_TIMEOUT_DEFAULT_S = 300
REMOTE_TIMEOUT_HEALTH_S = 5
REMOTE_TIMEOUT_ACTIVITY_S = 10
REMOTE_TIMEOUT_SCAN_S = 60 * 60
REMOTE_TIMEOUT_EXPORT_S = 60 * 60
REMOTE_TIMEOUT_TRAIN_S = 24 * 60 * 60

_MANAGED_PROCESS: subprocess.Popen | None = None
_MANAGED_API_BASE: str | None = None
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: dict = {
    "running": False,
    "operation": None,
    "stage": None,
    "message": "Idle",
    "target_mode": None,
    "api_base": None,
    "started_at": None,
    "updated_at": time.time(),
    "last_completed_at": None,
    "last_duration_s": None,
    "last_result_summary": None,
    "last_error": None,
}


@dataclass(frozen=True)
class BackendTarget:
    mode: str
    api_base: str | None = None


class ScanRequest(BaseModel):
    input_folder: str | None = None
    output_folder: str | None = None


class ExportRequest(BaseModel):
    output_folder: str | None = None
    min_images: int | None = Field(default=None, ge=1)
    identity_ids: list[int] | None = None


class TrainRequest(BaseModel):
    output_folder: str | None = None
    min_images: int | None = Field(default=None, ge=1)
    identity_ids: list[int] | None = None
    prepare_dataset: bool = False


class ResumeTrainingRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    prepare_dataset: bool = False


class RelabelRequest(BaseModel):
    identity_id: int = Field(..., ge=1)
    label: str | None = None


class MergeRequest(BaseModel):
    target_id: int = Field(..., ge=1)
    source_ids: list[int] = Field(..., min_length=1)


class SearchRequest(BaseModel):
    image_path: str
    top_k: int = Field(default=5, ge=1, le=100)


class ConnectionRequest(BaseModel):
    mode: str = "auto"
    api_base: str | None = None


class ImageActionRequest(BaseModel):
    image_path: str
    action: str = Field(..., description="remove | blacklist | restore")


class ReassignImageRequest(BaseModel):
    image_path: str
    identity_id: int = Field(..., ge=1)


class SwitchConfigRequest(BaseModel):
    config: str


class AutogenRequest(BaseModel):
    identity_id: int = Field(..., ge=1)
    target_count: int = Field(default=50, ge=1, le=10000)
    max_attempts: int = Field(default=500, ge=1, le=100000)
    assign_eps_realism: float | None = Field(default=None, ge=0.01, le=1.0)
    assign_eps_anime: float | None = Field(default=None, ge=0.01, le=1.0)
    target_identity_id: int | None = Field(default=None, ge=1)
    new_character_label: str | None = Field(default=None, min_length=1, max_length=100)


class UpdateConfigRequest(BaseModel):
    updates: dict


def get_router(_plugin_manifest: dict | None = None) -> APIRouter:
    router = APIRouter()

    @router.get("/connection")
    def get_connection() -> dict:
        config = _load_connection_config()
        target = _resolve_backend_target(start_managed=False)
        return _connection_payload(config=config, target=target)

    @router.post("/connection")
    def set_connection(payload: ConnectionRequest) -> dict:
        mode = str(payload.mode or "auto").strip().lower()
        if mode not in VALID_CONNECTION_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Use one of: {sorted(VALID_CONNECTION_MODES)}",
            )

        api_base = _normalize_api_base(payload.api_base)
        if mode == "remote_api" and not api_base:
            raise HTTPException(status_code=400, detail="api_base is required for remote_api mode.")

        config = {"mode": mode, "api_base": api_base or ""}
        _save_connection_config(config)

        # Let mode switching settle naturally; do not kill existing managed process.
        target = _resolve_backend_target(start_managed=False)
        return _connection_payload(config=config, target=target)

    @router.get("/health")
    def health() -> dict:
        target = _resolve_backend_target(start_managed=False)
        config = _load_connection_config()
        if target.api_base:
            payload = _remote_request(target.api_base, "GET", "/health")
            if isinstance(payload, dict):
                payload["_plugin_mode"] = target.mode
                payload["_remote_api_base"] = target.api_base
                payload["_connection_mode_config"] = config.get("mode", "auto")
                return payload
            return {
                "ok": False,
                "_plugin_mode": target.mode,
                "_remote_api_base": target.api_base,
                "_connection_mode_config": config.get("mode", "auto"),
                "raw": payload,
            }

        root = _resolve_dnaduck_root()
        return {
            "ok": root.exists(),
            "dnaduck_root": str(root),
            "dnaduck_config": str(_resolve_dnaduck_config(root)),
            "dnaduck_python": _resolve_python(),
            "_plugin_mode": "local_cli",
            "_connection_mode_config": config.get("mode", "auto"),
        }

    @router.get("/activity")
    def activity() -> dict:
        return _activity_snapshot()

    @router.get("/train-job-active")
    def train_job_active() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "GET",
                "/jobs/train/active",
                timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
            )
            return result if isinstance(result, dict) else {"job": None}
        return {"job": None}

    @router.get("/train-job/{job_id}")
    def train_job(job_id: str) -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "GET",
                f"/jobs/{urllib_parse.quote(str(job_id).strip())}",
                timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
            )
            return result if isinstance(result, dict) else {}
        raise HTTPException(status_code=400, detail="Train job lookup requires API mode.")

    @router.get("/train-jobs")
    def train_jobs(
        status: str | None = Query(default="paused"),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        local_jobs = _load_local_train_jobs(status=status, limit=limit)
        target = _resolve_backend_target(start_managed=False)
        if not target.api_base:
            return {"jobs": local_jobs}
        query = f"status={urllib_parse.quote(str(status or '').strip())}&limit={int(limit)}"
        try:
            result = _remote_request(
                target.api_base,
                "GET",
                f"/jobs/train?{query}",
                timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
            )
            remote_jobs = result.get("jobs", []) if isinstance(result, dict) else []
        except Exception:
            remote_jobs = []

        merged: dict[str, dict] = {}
        for row in remote_jobs:
            if not isinstance(row, dict):
                continue
            key = str(row.get("job_id") or "").strip()
            if not key:
                continue
            merged[key] = row
        for row in local_jobs:
            if not isinstance(row, dict):
                continue
            key = str(row.get("job_id") or "").strip()
            if not key or key in merged:
                continue
            merged[key] = row
        rows = list(merged.values())
        rows.sort(key=lambda item: _safe_float(item.get("created_at"), 0.0), reverse=True)
        return {"jobs": rows[: max(1, int(limit))]}

    @router.post("/resume-training")
    def resume_training(payload: ResumeTrainingRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if not target.api_base:
            raise HTTPException(status_code=400, detail="Resume is available only in API mode.")
        result = _remote_request(
            target.api_base,
            "POST",
            "/jobs/train/resume",
            {
                "job_id": str(payload.job_id).strip(),
                "prepare_dataset": bool(payload.prepare_dataset),
            },
            timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
        )
        data = result if isinstance(result, dict) else {}
        if bool(data.get("accepted")) and str(data.get("job_id", "")).strip():
            now = time.time()
            with _ACTIVITY_LOCK:
                _ACTIVITY["running"] = True
                _ACTIVITY["operation"] = "train_lora"
                _ACTIVITY["message"] = "Resumed training started in background."
                _ACTIVITY["stage"] = "queued"
                _ACTIVITY["started_at"] = now
                _ACTIVITY["updated_at"] = now
                _ACTIVITY["last_error"] = None
                _ACTIVITY["last_result_summary"] = _summarize_train_result(data)
        return {"result": data}

    @router.post("/scan")
    def scan(payload: ScanRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        _activity_start(
            operation="scan",
            target_mode=target.mode,
            api_base=target.api_base,
            message=(
                "Scan started. First run may download InsightFace models; "
                "network and RAM usage can spike."
            ),
        )
        try:
            if target.api_base:
                result = _remote_request(
                    target.api_base,
                    "POST",
                    "/scan",
                    {
                        "input_folder": payload.input_folder,
                        "output_folder": payload.output_folder,
                    },
                    timeout_s=REMOTE_TIMEOUT_SCAN_S,
                )
                summary = _summarize_scan_result(result)
                _activity_finish(
                    message="Scan complete.",
                    result_summary=summary,
                )
                return result

            args = ["scan"]
            if payload.input_folder:
                args.extend(["--input-folder", payload.input_folder])
            if payload.output_folder:
                args.extend(["--output-folder", payload.output_folder])
            stdout = _run_cli(args)
            result = {"result": _parse_colon_output(stdout), "raw": stdout}
            _activity_finish(
                message="Scan complete.",
                result_summary=_summarize_scan_result(result.get("result") or {}),
            )
            return result
        except HTTPException as exc:
            _activity_fail(_detail_to_text(exc.detail))
            raise
        except Exception as exc:
            _activity_fail(str(exc))
            raise

    @router.post("/scan-recluster")
    def scan_recluster(payload: ScanRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        _activity_start(
            operation="scan_recluster",
            target_mode=target.mode,
            api_base=target.api_base,
            message="Recluster-from-scratch started (DB reset + full rescan).",
        )
        try:
            if target.api_base:
                result = _remote_request(
                    target.api_base,
                    "POST",
                    "/scan/recluster",
                    {
                        "input_folder": payload.input_folder,
                        "output_folder": payload.output_folder,
                    },
                    timeout_s=REMOTE_TIMEOUT_SCAN_S,
                )
                summary = _summarize_scan_result(result)
                _activity_finish(
                    message="Recluster-from-scratch complete.",
                    result_summary=summary,
                )
                return result

            args = ["scan-recluster"]
            if payload.input_folder:
                args.extend(["--input-folder", payload.input_folder])
            if payload.output_folder:
                args.extend(["--output-folder", payload.output_folder])
            stdout = _run_cli(args)
            result = {"result": _parse_colon_output(stdout), "raw": stdout}
            _activity_finish(
                message="Recluster-from-scratch complete.",
                result_summary=_summarize_scan_result(result.get("result") or {}),
            )
            return result
        except HTTPException as exc:
            _activity_fail(_detail_to_text(exc.detail))
            raise
        except Exception as exc:
            _activity_fail(str(exc))
            raise

    @router.get("/identities")
    def identities(min_members: int = 1) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            rows = _remote_request(
                target.api_base,
                "GET",
                "/identities",
                query={"min_members": str(max(0, int(min_members)))},
            )
            return {"identities": rows, "_plugin_mode": target.mode}

        stdout = _run_cli(["identities", "--min-members", str(max(0, int(min_members)))])
        return {"identities": _parse_identities(stdout), "raw": stdout}

    @router.get("/identity/{identity_id}")
    def identity_detail(
        identity_id: int,
        limit: int = Query(default=120, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            return _remote_request(
                target.api_base,
                "GET",
                f"/identity/{int(identity_id)}",
                query={"limit": str(int(limit)), "offset": str(int(offset))},
            )

        stdout = _run_cli(
            [
                "identity",
                str(int(identity_id)),
                "--limit",
                str(int(limit)),
                "--offset",
                str(int(offset)),
            ]
        )
        detail = _parse_json_output(stdout)
        if not isinstance(detail, dict) or not detail:
            raise HTTPException(status_code=404, detail="Identity not found")
        return detail

    @router.post("/image/action")
    def image_action(payload: ImageActionRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "POST",
                "/image/action",
                {"image_path": payload.image_path, "action": payload.action},
            )
            return {"ok": bool(result.get("ok", False)), "result": result}

        stdout = _run_cli(
            [
                "image-action",
                str(payload.action),
                str(payload.image_path),
            ]
        )
        if stdout.strip().startswith("error:"):
            raise HTTPException(status_code=400, detail=stdout.strip())
        result = _parse_json_output(stdout)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="DNADuck image action returned invalid output.")
        return {"ok": bool(result.get("ok", False)), "result": result}

    @router.get("/image")
    def image(path: str = Query(...)) -> Response:
        normalized = _normalize_optional_path(path)
        if not normalized:
            raise HTTPException(status_code=400, detail="path is required")

        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            body, content_type = _remote_request_binary(
                target.api_base,
                "/image",
                query={"path": normalized},
            )
            return Response(content=body, media_type=content_type or "application/octet-stream")

        image_path = Path(normalized).resolve()
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        return FileResponse(str(image_path))

    @router.post("/image/reassign")
    def image_reassign(payload: ReassignImageRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            return _remote_request(
                target.api_base,
                "POST",
                "/image/reassign",
                {"image_path": payload.image_path, "identity_id": payload.identity_id},
            )
        stdout = _run_cli(
            ["reassign", str(payload.identity_id), str(payload.image_path)]
        )
        return _parse_json_output(stdout)

    @router.get("/images/unassigned")
    def images_unassigned(
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(
                target.api_base,
                "GET",
                "/images/unassigned",
                query={"limit": str(int(limit)), "offset": str(int(offset))},
            )
        stdout = _run_cli(
            ["list-unassigned", "--limit", str(int(limit)), "--offset", str(int(offset))]
        )
        return _parse_json_output(stdout)

    @router.get("/images/unassigned/count")
    def images_unassigned_count() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(
                target.api_base,
                "GET",
                "/images/unassigned/count",
            )
        stdout = _run_cli(["count-unassigned"])
        return _parse_json_output(stdout)

    @router.post("/images/unassigned/recluster")
    def images_unassigned_recluster() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(target.api_base, "POST", "/images/unassigned/recluster", {})
        stdout = _run_cli(["recluster-noise"])
        return _parse_json_output(stdout)

    @router.post("/images/unassigned/reanalyze")
    def images_unassigned_reanalyze() -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            return _remote_request(target.api_base, "POST", "/images/unassigned/reanalyze", {})
        stdout = _run_cli(["reanalyze-no-face"])
        return _parse_json_output(stdout)

    @router.post("/label")
    def label(payload: RelabelRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "POST",
                f"/identity/{int(payload.identity_id)}/label",
                {"label": payload.label},
            )
            return {"ok": bool(result.get("updated", False)), "raw": result}

        args = ["label", str(payload.identity_id)]
        if payload.label is not None:
            args.extend(["--text", payload.label])
        stdout = _run_cli(args)
        return {"ok": "updated=true" in stdout, "raw": stdout}

    @router.post("/merge")
    def merge(payload: MergeRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "POST",
                "/identity/merge",
                {
                    "target_id": int(payload.target_id),
                    "source_ids": [int(v) for v in payload.source_ids],
                },
            )
            return {"ok": bool(result.get("merged", False)), "raw": result}

        args = ["merge", str(payload.target_id)] + [str(v) for v in payload.source_ids]
        stdout = _run_cli(args)
        return {"ok": "merge_complete=true" in stdout, "raw": stdout}

    @router.post("/search")
    def search(payload: SearchRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        if target.api_base:
            result = _remote_request(
                target.api_base,
                "POST",
                "/search",
                {
                    "image_path": payload.image_path,
                    "top_k": int(payload.top_k),
                },
            )
            return {"matches": list(result.get("matches", [])), "raw": result}

        stdout = _run_cli(
            [
                "search",
                payload.image_path,
                "--top-k",
                str(int(payload.top_k)),
            ]
        )
        return {"matches": _parse_search(stdout), "raw": stdout}

    @router.post("/export-lora")
    def export_lora(payload: ExportRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        _activity_start(
            operation="export_lora",
            target_mode=target.mode,
            api_base=target.api_base,
            message="LoRA export started.",
        )
        try:
            if target.api_base:
                result = _remote_request(
                    target.api_base,
                    "POST",
                    "/export/lora",
                    {
                        "output_folder": payload.output_folder,
                        "min_images": payload.min_images,
                        "identity_ids": payload.identity_ids,
                    },
                    timeout_s=REMOTE_TIMEOUT_EXPORT_S,
                )
                _activity_finish(
                    message="LoRA export complete.",
                    result_summary=_summarize_export_result(result),
                )
                return {"result": result, "raw": result}

            args = ["export-lora"]
            if payload.output_folder:
                args.extend(["--output-folder", payload.output_folder])
            if payload.min_images is not None:
                args.extend(["--min-images", str(int(payload.min_images))])
            if payload.identity_ids:
                for identity_id in sorted({int(v) for v in payload.identity_ids if int(v) > 0}):
                    args.extend(["--identity-id", str(identity_id)])
            stdout = _run_cli(args)
            parsed = _parse_colon_output(stdout)
            _activity_finish(
                message="LoRA export complete.",
                result_summary=_summarize_export_result(parsed),
            )
            return {"result": parsed, "raw": stdout}
        except HTTPException as exc:
            _activity_fail(_detail_to_text(exc.detail))
            raise
        except Exception as exc:
            _activity_fail(str(exc))
            raise

    @router.post("/train-lora")
    def train_lora(payload: TrainRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        _activity_start(
            operation="train_lora",
            target_mode=target.mode,
            api_base=target.api_base,
            message="LoRA training command started.",
        )
        try:
            if target.api_base:
                result = _remote_request(
                    target.api_base,
                    "POST",
                    "/train/lora",
                    {
                        "output_folder": payload.output_folder,
                        "min_images": payload.min_images,
                        "identity_ids": payload.identity_ids,
                        "prepare_dataset": bool(payload.prepare_dataset),
                        "wait_for_result": False,
                    },
                    timeout_s=REMOTE_TIMEOUT_TRAIN_S,
                )
                if bool(result.get("accepted")) and str(result.get("job_id", "")).strip():
                    now = time.time()
                    with _ACTIVITY_LOCK:
                        _ACTIVITY["running"] = True
                        _ACTIVITY["operation"] = "train_lora"
                        _ACTIVITY["target_mode"] = target.mode
                        _ACTIVITY["api_base"] = target.api_base
                        _ACTIVITY["message"] = "Training started in background."
                        _ACTIVITY["stage"] = "queued"
                        _ACTIVITY["started_at"] = now
                        _ACTIVITY["updated_at"] = now
                        _ACTIVITY["last_error"] = None
                        _ACTIVITY["last_result_summary"] = _summarize_train_result(result)
                else:
                    _activity_finish(
                        message="LoRA training command finished.",
                        result_summary=_summarize_train_result(result),
                    )
                return {"result": result, "raw": result}

            export_result = None
            should_prepare = bool(payload.prepare_dataset or payload.min_images is not None or payload.identity_ids)
            if should_prepare:
                export_args = ["export-lora"]
                if payload.output_folder:
                    export_args.extend(["--output-folder", payload.output_folder])
                if payload.min_images is not None:
                    export_args.extend(["--min-images", str(int(payload.min_images))])
                if payload.identity_ids:
                    for identity_id in sorted({int(v) for v in payload.identity_ids if int(v) > 0}):
                        export_args.extend(["--identity-id", str(identity_id)])
                export_stdout = _run_cli(export_args)
                export_result = _parse_colon_output(export_stdout)

            train_args = ["train-lora"]
            if payload.output_folder:
                train_args.extend(["--output-folder", payload.output_folder])
            stdout = _run_cli(train_args)
            parsed = _parse_colon_output(stdout)
            if export_result is not None:
                parsed["prepared_dataset"] = True
                parsed["export_result"] = export_result
            else:
                parsed["prepared_dataset"] = False
            _activity_finish(
                message="LoRA training command finished.",
                result_summary=_summarize_train_result(parsed),
            )
            return {"result": parsed, "raw": stdout}
        except HTTPException as exc:
            _activity_fail(_detail_to_text(exc.detail))
            raise
        except Exception as exc:
            _activity_fail(str(exc))
            raise

    @router.post("/pause-training")
    def pause_training() -> dict:
        target = _resolve_backend_target(start_managed=True)
        if not target.api_base:
            raise HTTPException(
                status_code=400,
                detail="Pause is available only in API mode (managed_api or remote_api).",
            )
        result = _remote_request(
            target.api_base,
            "POST",
            "/jobs/train/pause",
            timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
        )
        data = result if isinstance(result, dict) else {}
        ok = bool(data.get("ok"))
        if ok:
            now = time.time()
            with _ACTIVITY_LOCK:
                _ACTIVITY["running"] = True
                _ACTIVITY["operation"] = "train_lora"
                _ACTIVITY["target_mode"] = target.mode
                _ACTIVITY["api_base"] = target.api_base
                _ACTIVITY["stage"] = "pausing"
                _ACTIVITY["message"] = str(data.get("message") or "Pause requested.")
                _ACTIVITY["updated_at"] = now
        return {"result": data}

    # ── Config endpoints ─────────────────────────────────────────────

    @router.get("/configs")
    def list_configs() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(target.api_base, "GET", "/configs", timeout_s=REMOTE_TIMEOUT_ACTIVITY_S)
        root = _resolve_dnaduck_root()
        cfg = _resolve_dnaduck_config(root)
        parent = cfg.parent.resolve()
        files = sorted(parent.glob("*.yaml")) + sorted(parent.glob("*.yml"))
        return {
            "configs": [_safe_rel_path(f, parent) for f in files],
            "current": _safe_rel_path(cfg, parent),
            "current_path": str(cfg),
            "root": str(parent),
        }

    def _safe_rel_path(path: Path, anchor: Path) -> str:
        try:
            return str(path.relative_to(anchor))
        except ValueError:
            return str(path)

    @router.get("/config")
    def get_config() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(target.api_base, "GET", "/config", timeout_s=REMOTE_TIMEOUT_ACTIVITY_S)
        root = _resolve_dnaduck_root()
        cfg = _resolve_dnaduck_config(root)
        try:
            raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            raw = {}
        return dict(raw)

    @router.post("/config/switch")
    def switch_config(payload: SwitchConfigRequest) -> dict:
        target = _resolve_backend_target(start_managed=False)
        if not target.api_base:
            raise HTTPException(status_code=400, detail="Config switching requires API mode (managed_api or remote_api).")
        return _remote_request(
            target.api_base,
            "POST",
            "/config/switch",
            {"config": payload.config},
            timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
        )

    @router.put("/config")
    def update_config(payload: UpdateConfigRequest) -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(
                target.api_base,
                "PUT",
                "/config",
                {"updates": payload.updates},
                timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
            )
        root = _resolve_dnaduck_root()
        cfg_path = _resolve_dnaduck_config(root)
        current: dict = {}
        try:
            current = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
        for key, value in payload.updates.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key].update(value)
            else:
                current[key] = value
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(current, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return {"ok": True, "updated": list(payload.updates.keys())}

    # ── Auto-generate endpoints ───────────────────────────────────────

    @router.post("/autogen/start")
    def autogen_start(payload: AutogenRequest) -> dict:
        target = _resolve_backend_target(start_managed=True)
        body = {
            "identity_id": int(payload.identity_id),
            "target_count": int(payload.target_count),
            "max_attempts": int(payload.max_attempts),
        }
        if payload.assign_eps_realism is not None:
            body["assign_eps_realism"] = float(payload.assign_eps_realism)
        if payload.assign_eps_anime is not None:
            body["assign_eps_anime"] = float(payload.assign_eps_anime)
        if payload.target_identity_id is not None:
            body["target_identity_id"] = int(payload.target_identity_id)
        if payload.new_character_label is not None:
            body["new_character_label"] = payload.new_character_label.strip()
        if target.api_base:
            return _remote_request(
                target.api_base,
                "POST",
                "/autogen/start",
                body,
                timeout_s=REMOTE_TIMEOUT_SCAN_S,
            )
        cli_cmd = [
            "autogen",
            str(int(payload.identity_id)),
            "--target-count", str(int(payload.target_count)),
            "--max-attempts", str(int(payload.max_attempts)),
        ]
        if payload.assign_eps_realism is not None:
            cli_cmd += ["--assign-eps-realism", str(payload.assign_eps_realism)]
        if payload.assign_eps_anime is not None:
            cli_cmd += ["--assign-eps-anime", str(payload.assign_eps_anime)]
        stdout = _run_cli(cli_cmd)
        return _parse_json_output(stdout)

    @router.post("/autogen/cancel")
    def autogen_cancel() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(target.api_base, "POST", "/autogen/cancel", {})
        stdout = _run_cli(["autogen-cancel"])
        return _parse_json_output(stdout)

    @router.get("/autogen/status")
    def autogen_status() -> dict:
        target = _resolve_backend_target(start_managed=False)
        if target.api_base:
            return _remote_request(target.api_base, "GET", "/autogen/status")
        stdout = _run_cli(["autogen-status"])
        return _parse_json_output(stdout)

    return router


def _activity_start(*, operation: str, target_mode: str, api_base: str | None, message: str) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        _ACTIVITY["running"] = True
        _ACTIVITY["operation"] = operation
        _ACTIVITY["stage"] = "starting"
        _ACTIVITY["target_mode"] = target_mode
        _ACTIVITY["api_base"] = api_base
        _ACTIVITY["message"] = message
        _ACTIVITY["started_at"] = now
        _ACTIVITY["updated_at"] = now
        _ACTIVITY["last_error"] = None
        _ACTIVITY["last_result_summary"] = None


def _activity_finish(*, message: str, result_summary: dict | None = None) -> None:
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
        _ACTIVITY["last_result_summary"] = result_summary
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

    api_base = str(payload.get("api_base") or "").strip()
    if not api_base:
        try:
            target = _resolve_backend_target(start_managed=False)
            if target.api_base:
                api_base = str(target.api_base).strip()
                payload["api_base"] = api_base
                payload["target_mode"] = target.mode
        except Exception:
            api_base = ""

    if api_base:
        try:
            remote_activity = _remote_request(
                api_base,
                "GET",
                "/activity",
                timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
            )
            if isinstance(remote_activity, dict):
                payload["remote_activity"] = remote_activity
                # Remote API is authoritative when present.
                if "running" in remote_activity:
                    payload["running"] = bool(remote_activity.get("running"))
                for key in (
                    "message",
                    "operation",
                    "stage",
                    "started_at",
                    "updated_at",
                    "last_completed_at",
                    "last_duration_s",
                    "last_error",
                    "elapsed_s",
                ):
                    if key in remote_activity and remote_activity.get(key) is not None:
                        payload[key] = remote_activity.get(key)

                # Clear stale local "queued/pausing" state when remote has no active train job.
                if not bool(remote_activity.get("running")):
                    try:
                        active_job_payload = _remote_request(
                            api_base,
                            "GET",
                            "/jobs/train/active",
                            timeout_s=REMOTE_TIMEOUT_ACTIVITY_S,
                        )
                        active_job = (
                            active_job_payload.get("job")
                            if isinstance(active_job_payload, dict)
                            else None
                        )
                    except Exception:
                        active_job = None
                    stage_text = str(payload.get("stage") or "").strip().lower()
                    if active_job is None and (
                        stage_text in {"queued", "pausing", "training", "training_preparing", "training_starting"}
                        or stage_text.startswith("training")
                    ):
                        payload["running"] = False
                        payload["stage"] = "complete"
                        payload["message"] = "No active training job."
        except Exception as exc:
            # If remote status cannot be read, do not keep stale "running" forever.
            payload["running"] = False
            payload["stage"] = "failed"
            payload["message"] = "Remote activity unavailable."
            payload["last_error"] = str(exc)[:4000]

    started_at = payload.get("started_at")
    if payload.get("elapsed_s") is None:
        payload["elapsed_s"] = (
            max(0.0, now - float(started_at))
            if payload.get("running") and isinstance(started_at, (int, float))
            else None
        )
    return payload


def _detail_to_text(detail: object) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, ensure_ascii=True)
    except Exception:
        return str(detail)


def _summarize_scan_result(payload: dict | list | None) -> dict:
    data = payload if isinstance(payload, dict) else {}
    return {
        "discovered_count": data.get("discovered_count"),
        "processed_count": data.get("processed_count"),
        "assigned_count": data.get("assigned_count"),
        "noise_count": data.get("noise_count"),
        "no_face_count": data.get("no_face_count"),
    }


def _summarize_export_result(payload: dict | list | None) -> dict:
    data = payload if isinstance(payload, dict) else {}
    return {
        "identities_exported": data.get("identities_exported"),
        "images_exported": data.get("images_exported"),
        "caption_mode": data.get("caption_mode"),
        "requested_identity_ids": data.get("requested_identity_ids"),
    }


def _summarize_train_result(payload: dict | list | None) -> dict:
    data = payload if isinstance(payload, dict) else {}
    export_result = data.get("export_result") if isinstance(data.get("export_result"), dict) else {}
    return {
        "accepted": data.get("accepted"),
        "job_id": data.get("job_id"),
        "status": data.get("status"),
        "stopped_by_user": data.get("stopped_by_user"),
        "returncode": data.get("returncode"),
        "dataset_dir": data.get("dataset_dir"),
        "output_dir": data.get("output_dir"),
        "log_file": data.get("log_file"),
        "artifacts_after": data.get("artifacts_after"),
        "new_artifacts_count": len(data.get("new_artifacts", [])) if isinstance(data.get("new_artifacts"), list) else None,
        "prepared_dataset": data.get("prepared_dataset"),
        "identities_exported": export_result.get("identities_exported"),
    }


def _load_local_train_jobs(*, status: str | None = None, limit: int = 50) -> list[dict]:
    try:
        root = _resolve_dnaduck_root()
        cfg = _resolve_dnaduck_config(root)
        state_path = cfg.parent / ".dnaduck_train_jobs.json"
        if not state_path.exists():
            return []
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
        rows = raw_payload.get("jobs") if isinstance(raw_payload, dict) else []
        if not isinstance(rows, list):
            return []
        statuses = _normalize_status_set(status)
        jobs: list[dict] = []
        for row in rows:
            summary = _summarize_local_train_job(row)
            if not summary:
                continue
            if statuses and str(summary.get("status", "")).lower() not in statuses:
                continue
            jobs.append(summary)
        jobs.sort(key=lambda item: float(item.get("created_at") or 0.0), reverse=True)
        return jobs[: max(1, int(limit))]
    except Exception:
        return []


def _normalize_status_set(value: str | None) -> set[str] | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summarize_local_train_job(raw: object) -> dict:
    if not isinstance(raw, dict):
        return {}
    request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    raw_ids = request.get("identity_ids") if isinstance(request.get("identity_ids"), list) else []
    identity_ids: list[int] = []
    for value in raw_ids:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            identity_ids.append(parsed)
    created_at = raw.get("created_at")
    try:
        created_at = float(created_at)
    except (TypeError, ValueError):
        created_at = 0.0
    return {
        "job_id": raw.get("job_id"),
        "status": raw.get("status"),
        "operation": raw.get("operation"),
        "created_at": created_at,
        "updated_at": raw.get("updated_at"),
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "cancel_requested": bool(raw.get("cancel_requested")),
        "resumed_from_job_id": raw.get("resumed_from_job_id"),
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
            "new_artifacts_count": len(result.get("new_artifacts", []))
            if isinstance(result.get("new_artifacts"), list)
            else None,
        },
        "error": raw.get("error"),
    }


def _connection_payload(config: dict, target: BackendTarget) -> dict:
    return {
        "mode": config.get("mode", "auto"),
        "api_base": config.get("api_base", ""),
        "effective_mode": target.mode,
        "effective_api_base": target.api_base,
        "managed_running": bool(_MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is None),
        "dnaduck_root": str(_resolve_dnaduck_root()),
        "dnaduck_config": str(_resolve_dnaduck_config(_resolve_dnaduck_root())),
    }


def _resolve_python() -> str:
    return os.environ.get("DNADUCK_PYTHON", sys.executable)


def _load_connection_config() -> dict:
    config = {"mode": "auto", "api_base": ""}
    if STATE_FILE.exists():
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                mode = str(raw.get("mode", "auto")).strip().lower()
                if mode in VALID_CONNECTION_MODES:
                    config["mode"] = mode
                config["api_base"] = _normalize_api_base(raw.get("api_base")) or ""
        except Exception:
            pass

    env_mode = str(os.environ.get("DNADUCK_CONNECTION_MODE", "")).strip().lower()
    if env_mode in VALID_CONNECTION_MODES:
        config["mode"] = env_mode

    env_base = _normalize_api_base(os.environ.get("DNADUCK_API_BASE"))
    if env_base:
        config["mode"] = "remote_api"
        config["api_base"] = env_base
    return config


def _save_connection_config(config: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _normalize_api_base(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urllib_parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return raw.rstrip("/")


def _normalize_optional_path(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw or None


def _resolve_backend_target(start_managed: bool) -> BackendTarget:
    config = _load_connection_config()
    mode = str(config.get("mode", "auto")).strip().lower()
    api_base = _normalize_api_base(config.get("api_base"))

    if mode == "remote_api":
        if not api_base:
            raise HTTPException(status_code=400, detail="Connection mode is remote_api but api_base is not set.")
        return BackendTarget(mode="remote_api", api_base=api_base)

    if mode == "managed_api":
        managed = _ensure_managed_api(start=start_managed)
        if not managed:
            if not start_managed:
                return BackendTarget(mode="managed_api", api_base=None)
            raise HTTPException(status_code=500, detail="Managed DNADuck API could not be started.")
        return BackendTarget(mode="managed_api", api_base=managed)

    if mode == "local_cli":
        return BackendTarget(mode="local_cli", api_base=None)

    # auto mode:
    managed = _ensure_managed_api(start=start_managed)
    if managed:
        return BackendTarget(mode="managed_api", api_base=managed)
    return BackendTarget(mode="local_cli", api_base=None)


def _ensure_managed_api(start: bool) -> str | None:
    global _MANAGED_PROCESS, _MANAGED_API_BASE

    if _MANAGED_PROCESS is not None and _MANAGED_PROCESS.poll() is not None:
        _MANAGED_PROCESS = None
        _MANAGED_API_BASE = None

    if _MANAGED_API_BASE and _remote_health_ok(_MANAGED_API_BASE):
        return _MANAGED_API_BASE

    root = _resolve_dnaduck_root()
    run_api = root / "run_api.py"
    if not run_api.exists():
        return None

    host = str(os.environ.get("DNADUCK_MANAGED_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    preferred_port = int(str(os.environ.get("DNADUCK_MANAGED_PORT", "8020")).strip() or "8020")
    candidate_ports = [preferred_port, 8020, 8025]
    seen_ports: set[int] = set()
    ordered_ports: list[int] = []
    for value in candidate_ports:
        if int(value) in seen_ports:
            continue
        seen_ports.add(int(value))
        ordered_ports.append(int(value))

    # Reuse already-running external DNADuck API if it's healthy.
    for port in ordered_ports:
        base = f"http://{host}:{port}"
        if _remote_health_ok(base):
            _MANAGED_API_BASE = base
            return base

    if not start:
        return None

    launch_port = ordered_ports[0]
    launch_base = f"http://{host}:{launch_port}"
    config_path = _resolve_dnaduck_config(root)
    cmd = [
        _resolve_python(),
        str(run_api),
        "--host",
        host,
        "--port",
        str(launch_port),
        "--config",
        str(config_path),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):
        if proc.poll() is not None:
            break
        if _remote_health_ok(launch_base):
            _MANAGED_PROCESS = proc
            _MANAGED_API_BASE = launch_base
            return launch_base
        time.sleep(0.25)

    try:
        proc.terminate()
    except Exception:
        pass
    return None


def _remote_health_ok(base: str) -> bool:
    try:
        _remote_request(base, "GET", "/health", timeout_s=REMOTE_TIMEOUT_HEALTH_S)
        return True
    except Exception:
        return False


def _remote_request(
    base: str,
    method: str,
    path: str,
    payload: dict | None = None,
    query: dict[str, str] | None = None,
    timeout_s: int | float = REMOTE_TIMEOUT_DEFAULT_S,
) -> dict | list:
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib_parse.urlencode(query)}"

    data_bytes = None
    headers = {}
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url=url, data=data_bytes, method=method.upper(), headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=float(timeout_s)) as response:
            body = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return json.loads(body) if body else {}
            return {"raw": body}
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail: object
        parsed_body: object = None
        if body:
            try:
                parsed_body = json.loads(body)
            except Exception:
                parsed_body = None
        if isinstance(parsed_body, dict):
            if "detail" in parsed_body:
                detail = parsed_body.get("detail")
            elif "error" in parsed_body or "message" in parsed_body:
                detail = parsed_body
            else:
                detail = {
                    "error": "Remote DNADuck API request failed",
                    "url": url,
                    "status": exc.code,
                    "body": body[-4000:],
                }
        elif isinstance(parsed_body, list):
            detail = parsed_body
        else:
            detail = {
                "error": "Remote DNADuck API request failed",
                "url": url,
                "status": exc.code,
                "body": body[-4000:],
            }
        raise HTTPException(
            status_code=int(exc.code) if int(exc.code) > 0 else 500,
            detail=detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Remote DNADuck API request failed",
                "url": url,
                "message": str(exc),
            },
        ) from exc


def _remote_request_binary(
    base: str,
    path: str,
    *,
    query: dict[str, str] | None = None,
) -> tuple[bytes, str | None]:
    url = f"{base}{path}"
    if query:
        url = f"{url}?{urllib_parse.urlencode(query)}"
    req = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=300) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type")
            return body, content_type
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail: object
        parsed_body: object = None
        if body:
            try:
                parsed_body = json.loads(body)
            except Exception:
                parsed_body = None
        if isinstance(parsed_body, dict):
            detail = parsed_body.get("detail", parsed_body)
        elif isinstance(parsed_body, list):
            detail = parsed_body
        else:
            detail = {
                "error": "Remote DNADuck API request failed",
                "url": url,
                "status": exc.code,
                "body": body[-4000:],
            }
        raise HTTPException(
            status_code=int(exc.code) if int(exc.code) > 0 else 500,
            detail=detail,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Remote DNADuck API request failed",
                "url": url,
                "message": str(exc),
            },
        ) from exc


def _resolve_dnaduck_root() -> Path:
    env_path = os.environ.get("DNADUCK_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates: list[Path] = []

    # Case 1: embedded in plugin folder.
    candidates.append(PLUGIN_ROOT / "dnaduck")

    # Typical sibling layout: /img-gen/webbduck + /img-gen/dnaduck
    plugin_parents = list(PLUGIN_ROOT.parents)
    if len(plugin_parents) >= 3:
        # .../webbduck/plugins/webapps/dnaduck -> parents[2] == .../webbduck
        candidates.append(plugin_parents[2].parent / "dnaduck")
    if len(plugin_parents) >= 4:
        # .../dnaduck/integrations/webbduck_plugin/webapps/dnaduck -> parents[3] == .../dnaduck
        candidates.append(plugin_parents[3])

    # Alternate: plugin directory copied elsewhere.
    candidates.extend(
        [
            Path.cwd() / "dnaduck",
            Path.cwd().parent / "dnaduck",
            PLUGIN_ROOT.parent / "dnaduck",
        ]
    )
    for candidate in candidates:
        if (candidate / "main.py").exists() and (candidate / "core").exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_dnaduck_config(dnaduck_root: Path) -> Path:
    env_cfg = os.environ.get("DNADUCK_CONFIG")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    return (dnaduck_root / "config.yaml").resolve()


def _run_cli(args: list[str]) -> str:
    dnaduck_root = _resolve_dnaduck_root()
    config_path = _resolve_dnaduck_config(dnaduck_root)
    main_path = dnaduck_root / "main.py"

    if not main_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"DNADuck main.py not found at: {main_path}",
        )

    cmd = [_resolve_python(), str(main_path), "--config", str(config_path), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(dnaduck_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        hint = None
        stderr = (proc.stderr or "")[-4000:]
        if "No module named" in stderr:
            hint = (
                "DNADuck dependencies are missing in WebbDuck's Python env. "
                "Use remote_api mode pointing at a running DNADuck API, or set DNADUCK_PYTHON "
                "to a Python environment where DNADuck requirements are installed."
            )
        elif "Input folder does not exist" in stderr or "No images found in" in stderr:
            hint = (
                "Set a valid Input Folder in DNADuck UI Scan section, "
                "or update dnaduck/config.yaml input_folder."
            )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DNADuck command failed",
                "command": cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": stderr,
                "hint": hint,
            },
        )
    return (proc.stdout or "").strip()


def _parse_colon_output(stdout: str) -> dict:
    out: dict[str, object] = {}
    for line in stdout.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            out[key] = value.lower() == "true"
            continue
        if value.startswith("[") or value.startswith("{"):
            try:
                out[key] = json.loads(value)
                continue
            except Exception:
                pass
        if re.fullmatch(r"-?\d+", value):
            out[key] = int(value)
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def _parse_json_output(stdout: str) -> dict | list:
    raw = str(stdout or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DNADuck command returned non-JSON output.",
                "stdout": raw[-2000:],
                "message": str(exc),
            },
        ) from exc


def _parse_identities(stdout: str) -> list[dict]:
    identities: list[dict] = []
    pattern = re.compile(
        r"^id=(?P<id>\d+)\s+members=(?P<members>\d+)\s+label=(?P<label>.*?)\s+updated=(?P<updated>.+)$"
    )
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("total="):
            continue
        match = pattern.match(line)
        if not match:
            continue
        label = match.group("label")
        identities.append(
            {
                "identity_id": int(match.group("id")),
                "member_count": int(match.group("members")),
                "label": None if label == "None" else label,
                "updated_at": match.group("updated"),
            }
        )
    return identities


def _parse_search(stdout: str) -> list[dict]:
    rows: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("total="):
            continue
        if not line.startswith("identity_id="):
            continue
        parts = line.split()
        parsed: dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key] = value
        rows.append(
            {
                "identity_id": int(parsed.get("identity_id", "0")),
                "similarity": float(parsed.get("similarity", "0")),
                "distance": float(parsed.get("distance", "1")),
                "member_count": int(parsed.get("members", "0")),
                "label": None if parsed.get("label") in {None, "None"} else parsed.get("label"),
            }
        )
    return rows
