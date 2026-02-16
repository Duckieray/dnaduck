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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

PLUGIN_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path.home() / ".webbduck" / "plugin_state"
STATE_FILE = STATE_DIR / "dnaduck_connection.json"
VALID_CONNECTION_MODES = {"auto", "local_cli", "remote_api", "managed_api"}

_MANAGED_PROCESS: subprocess.Popen | None = None
_MANAGED_API_BASE: str | None = None
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: dict = {
    "running": False,
    "operation": None,
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
        target = _resolve_backend_target(start_managed=True)
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
                    },
                )
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

    return router


def _activity_start(*, operation: str, target_mode: str, api_base: str | None, message: str) -> None:
    now = time.time()
    with _ACTIVITY_LOCK:
        _ACTIVITY["running"] = True
        _ACTIVITY["operation"] = operation
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
        _ACTIVITY["message"] = "Operation failed."
        _ACTIVITY["updated_at"] = now
        _ACTIVITY["last_completed_at"] = now
        _ACTIVITY["last_error"] = str(error_text)[:4000]


def _activity_snapshot() -> dict:
    now = time.time()
    with _ACTIVITY_LOCK:
        payload = dict(_ACTIVITY)

    api_base = str(payload.get("api_base") or "").strip()
    if api_base:
        try:
            remote_activity = _remote_request(api_base, "GET", "/activity")
            if isinstance(remote_activity, dict):
                payload["remote_activity"] = remote_activity
                if bool(remote_activity.get("running", False)):
                    payload["running"] = True
                if remote_activity.get("message"):
                    payload["message"] = remote_activity.get("message")
                if remote_activity.get("operation"):
                    payload["operation"] = remote_activity.get("operation")
                if remote_activity.get("stage"):
                    payload["stage"] = remote_activity.get("stage")
        except Exception:
            pass

    started_at = payload.get("started_at")
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
        "returncode": data.get("returncode"),
        "dataset_dir": data.get("dataset_dir"),
        "prepared_dataset": data.get("prepared_dataset"),
        "identities_exported": export_result.get("identities_exported"),
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
        _remote_request(base, "GET", "/health")
        return True
    except Exception:
        return False


def _remote_request(
    base: str,
    method: str,
    path: str,
    payload: dict | None = None,
    query: dict[str, str] | None = None,
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
        with urllib_request.urlopen(req, timeout=300) as response:
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
