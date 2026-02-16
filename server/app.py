from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.service import (
    apply_image_action,
    export_lora,
    get_identities,
    get_identity_detail,
    merge_identity_groups,
    relabel_identity,
    scan_images,
    scan_recluster_from_scratch,
    search_by_image,
    trigger_lora_training,
)


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


class ImageActionRequest(BaseModel):
    image_path: str
    action: str = Field(..., description="remove | blacklist | restore")


_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: dict = {
    "running": False,
    "operation": None,
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
    "last_result": None,
    "last_error": None,
}


def _normalize_optional_path(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1].strip()
    return raw or None


def _activity_start(operation: str, message: str) -> None:
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
        ):
            _ACTIVITY[key] = None
        _ACTIVITY["running"] = True
        _ACTIVITY["operation"] = operation
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
        _ACTIVITY["message"] = "Scan failed."
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


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="DNADuck API", version="0.1.0")
    cfg_path = Path(config_path or os.environ.get("DNADUCK_CONFIG", "config.yaml")).resolve()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "config_path": str(cfg_path)}

    @app.get("/activity")
    def activity() -> dict:
        return _activity_snapshot()

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
                config_path=cfg_path,
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
                config_path=cfg_path,
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
        return get_identities(config_path=cfg_path, min_members=min_members)

    @app.get("/identity/{identity_id}")
    def identity_detail(
        identity_id: int,
        limit: int = Query(default=120, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        data = get_identity_detail(
            config_path=cfg_path,
            identity_id=identity_id,
            limit=limit,
            offset=offset,
        )
        if not data:
            raise HTTPException(status_code=404, detail="Identity not found")
        return data

    @app.post("/identity/{identity_id}/label")
    def identity_label(identity_id: int, payload: RelabelRequest) -> dict:
        updated = relabel_identity(config_path=cfg_path, identity_id=identity_id, label=payload.label)
        if not updated:
            raise HTTPException(status_code=404, detail="Identity not found")
        return {"updated": True, "identity_id": identity_id, "label": payload.label}

    @app.post("/identity/merge")
    def identity_merge(payload: MergeRequest) -> dict:
        merge_identity_groups(
            config_path=cfg_path,
            target_id=payload.target_id,
            source_ids=payload.source_ids,
        )
        return {"merged": True, "target_id": payload.target_id, "source_ids": payload.source_ids}

    @app.post("/search")
    def search(payload: SearchRequest) -> dict:
        try:
            rows = search_by_image(
                config_path=cfg_path,
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
        overrides = {"output_folder": _normalize_optional_path(payload.output_folder)}
        if payload.min_images is not None:
            overrides["lora_min_images"] = int(payload.min_images)
        if payload.identity_ids:
            overrides["lora_identity_ids"] = [int(v) for v in payload.identity_ids if int(v) > 0]
        try:
            return export_lora(config_path=cfg_path, overrides=overrides)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LoRA export failed: {exc}") from exc

    @app.post("/train/lora")
    def train_lora_endpoint(payload: LoraTrainRequest) -> dict:
        try:
            export_result: dict | None = None
            if payload.prepare_dataset or payload.min_images is not None or payload.identity_ids:
                export_overrides = {"output_folder": _normalize_optional_path(payload.output_folder)}
                if payload.min_images is not None:
                    export_overrides["lora_min_images"] = int(payload.min_images)
                if payload.identity_ids:
                    export_overrides["lora_identity_ids"] = [int(v) for v in payload.identity_ids if int(v) > 0]
                export_result = export_lora(config_path=cfg_path, overrides=export_overrides)

            result = trigger_lora_training(
                config_path=cfg_path,
                overrides={"output_folder": _normalize_optional_path(payload.output_folder)},
            )
            if export_result is not None:
                result["prepared_dataset"] = True
                result["export_result"] = export_result
            else:
                result["prepared_dataset"] = False
            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LoRA training failed: {exc}") from exc

    @app.post("/image/action")
    def image_action(payload: ImageActionRequest) -> dict:
        image_path = _normalize_optional_path(payload.image_path)
        if not image_path:
            raise HTTPException(status_code=400, detail="image_path is required")
        try:
            return apply_image_action(
                config_path=cfg_path,
                image_path=Path(image_path),
                action=payload.action,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image action failed: {exc}") from exc

    @app.get("/image")
    def image(path: str = Query(...)) -> FileResponse:
        normalized = _normalize_optional_path(path)
        if not normalized:
            raise HTTPException(status_code=400, detail="path is required")
        image_path = Path(normalized).resolve()
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        return FileResponse(str(image_path))

    return app


app = create_app()
