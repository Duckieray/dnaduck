from __future__ import annotations
import shlex
import subprocess
import sys
from pathlib import Path

from .database import (
    compute_identity_drift,
    count_identity_images,
    fetch_image_by_path,
    list_all_images,
    list_identities,
    list_identity_images_page,
    merge_identities,
    open_database,
    rebuild_identity_stats,
    update_image_status,
    update_identity_label,
)
from .pipeline import run_lora_export, run_scan, search_identity_candidates
from .utils import load_config, resolve_runtime_paths


def load_runtime_config(config_path: Path, overrides: dict | None = None) -> dict:
    config = load_config(config_path.resolve())
    overrides = overrides or {}
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config = resolve_runtime_paths(config, config_path.resolve())
    return config


def scan_images(
    config_path: Path,
    overrides: dict | None = None,
    progress_callback=None,
) -> dict:
    config = load_runtime_config(config_path=config_path, overrides=overrides)
    return _run_scan_with_config(config, progress_callback=progress_callback)


def scan_recluster_from_scratch(
    config_path: Path,
    overrides: dict | None = None,
    progress_callback=None,
) -> dict:
    config = load_runtime_config(config_path=config_path, overrides=overrides)
    _reset_database_files(Path(config["database_path"]))
    result = _run_scan_with_config(config, progress_callback=progress_callback)
    result["recluster_from_scratch"] = True
    return result


def _run_scan_with_config(config: dict, progress_callback=None) -> dict:
    result = run_scan(config, progress_callback=progress_callback)
    return {
        "scan_id": result.scan_id,
        "discovered_count": result.discovered_count,
        "processed_count": result.processed_count,
        "assigned_count": result.assigned_count,
        "noise_count": result.noise_count,
        "no_face_count": result.no_face_count,
        "identity_count": result.identity_count,
        "metadata_images": result.metadata_images,
        "metadata_identities": result.metadata_identities,
        "linked_images": result.linked_images,
        "database_path": str(config["database_path"]),
        "output_folder": str(config["output_folder"]),
    }


def _reset_database_files(db_path: Path) -> None:
    db_path = db_path.resolve()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{db_path}{suffix}")
        try:
            if candidate.exists():
                candidate.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Best effort cleanup for SQLite side files.
            pass


def export_lora(config_path: Path, overrides: dict | None = None) -> dict:
    config = load_runtime_config(config_path=config_path, overrides=overrides)
    result = run_lora_export(config)
    return {
        "identities_exported": int(result["identities"]),
        "images_exported": int(result["images"]),
        "caption_mode": str(result.get("caption_mode", "identity_token")),
        "requested_identity_ids": list(result.get("requested_identity_ids", [])),
        "exported_identity_ids": list(result.get("exported_identity_ids", [])),
        "rich_captioning_status": "coming_soon",
        "output_folder": str(Path(config["output_folder"]).resolve() / "lora_export"),
    }


def get_identities(config_path: Path, min_members: int = 1) -> list[dict]:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        rows = list_identities(conn, min_members=min_members)
        return [
            {
                "identity_id": int(row["id"]),
                "label": row["label"],
                "member_count": int(row["member_count"]),
                "drift_score": compute_identity_drift(conn, int(row["id"])),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_identity_detail(config_path: Path, identity_id: int, *, limit: int = 120, offset: int = 0) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        identity_rows = [row for row in list_identities(conn, min_members=0) if int(row["id"]) == int(identity_id)]
        if not identity_rows:
            return {}
        row = identity_rows[0]
        total = count_identity_images(conn, identity_id=int(identity_id))
        page = list_identity_images_page(
            conn,
            identity_id=int(identity_id),
            limit=max(1, int(limit)),
            offset=max(0, int(offset)),
        )
        return {
            "identity_id": int(row["id"]),
            "label": row["label"],
            "member_count": int(row["member_count"]),
            "drift_score": compute_identity_drift(conn, int(identity_id)),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "images_total": int(total),
            "images_offset": max(0, int(offset)),
            "images_limit": max(1, int(limit)),
            "images": [
                {
                    "path": str(image_row["path"]),
                    "status": str(image_row["status"]),
                    "exists": Path(str(image_row["path"])).exists(),
                }
                for image_row in page
            ],
        }
    finally:
        conn.close()


def relabel_identity(config_path: Path, identity_id: int, label: str | None) -> bool:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        return update_identity_label(conn, identity_id=identity_id, label=label)
    finally:
        conn.close()


def merge_identity_groups(config_path: Path, target_id: int, source_ids: list[int]) -> None:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        merge_identities(conn, target_id=target_id, source_ids=source_ids)
    finally:
        conn.close()


def search_by_image(config_path: Path, image_path: Path, top_k: int = 5) -> list[dict]:
    config = load_runtime_config(config_path=config_path)
    return search_identity_candidates(config, image_path=image_path, top_k=top_k)


def list_images(config_path: Path) -> list[dict]:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        rows = list_all_images(conn)
        return [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "status": row["status"],
                "identity_id": row["identity_id"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_seen_at": row["last_seen_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def apply_image_action(config_path: Path, image_path: Path, action: str) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    resolved = image_path.resolve()
    normalized_action = str(action).strip().lower()
    try:
        before = fetch_image_by_path(conn, resolved)
        if before is None:
            raise FileNotFoundError(f"Image is not tracked in DNADuck DB: {resolved}")

        if normalized_action == "remove":
            ok = update_image_status(
                conn,
                path=resolved,
                status="removed",
                identity_id=None,
                clear_embedding=False,
            )
        elif normalized_action == "blacklist":
            ok = update_image_status(
                conn,
                path=resolved,
                status="blacklisted",
                identity_id=None,
                clear_embedding=True,
            )
        elif normalized_action == "restore":
            ok = update_image_status(
                conn,
                path=resolved,
                status="noise",
                identity_id=None,
                clear_embedding=False,
            )
        else:
            raise ValueError("Unsupported image action. Expected: remove | blacklist | restore")

        if not ok:
            raise FileNotFoundError(f"Image is not tracked in DNADuck DB: {resolved}")

        rebuild_identity_stats(conn)
        after = fetch_image_by_path(conn, resolved)
        return {
            "ok": True,
            "action": normalized_action,
            "image_path": str(resolved),
            "before_status": None if before is None else str(before["status"]),
            "before_identity_id": None if before is None else before["identity_id"],
            "after_status": None if after is None else str(after["status"]),
            "after_identity_id": None if after is None else after["identity_id"],
        }
    finally:
        conn.close()


def trigger_lora_training(config_path: Path, overrides: dict | None = None) -> dict:
    config = load_runtime_config(config_path=config_path, overrides=overrides)
    dataset_dir = Path(config["output_folder"]).resolve() / "lora_export"
    command_template = str(config.get("lora_train_command", "")).strip()
    trainer = str(config.get("lora_trainer", "kohya_ss")).strip().lower()

    if command_template:
        command = shlex.split(command_template.format(dataset_dir=str(dataset_dir)))
        return _run_command(command, dataset_dir=dataset_dir)

    if trainer == "kohya_ss":
        command = _build_kohya_command(config, dataset_dir=dataset_dir)
        return _run_command(command, dataset_dir=dataset_dir)

    raise ValueError(
        "No supported training configuration found. "
        "Set lora_train_command or lora_trainer='kohya_ss' in config."
    )


def _build_kohya_command(config: dict, dataset_dir: Path) -> list[str]:
    dnaduck_root = Path(__file__).resolve().parent.parent
    tool_path = dnaduck_root / "tools" / "train_kohya_lora.py"
    sd_scripts_dir = str(config.get("kohya_sd_scripts_dir", "")).strip()
    base_model = str(config.get("kohya_base_model", "")).strip()
    output_dir = str(config.get("kohya_output_dir", "")).strip() or str(
        (Path(config["output_folder"]).resolve() / "trained_loras")
    )
    output_name = str(config.get("kohya_output_name", "dnaduck_lora")).strip()
    steps = int(config.get("kohya_train_steps", 1000))
    learning_rate = float(config.get("kohya_learning_rate", 1e-4))
    network_dim = int(config.get("kohya_network_dim", 32))
    network_alpha = int(config.get("kohya_network_alpha", 16))
    batch_size = int(config.get("kohya_batch_size", 1))
    resolution = str(config.get("kohya_resolution", "1024,1024"))
    repeats = int(config.get("kohya_num_repeats", 10))

    if not sd_scripts_dir:
        raise ValueError("kohya_sd_scripts_dir is not set in config.")
    if not base_model:
        raise ValueError("kohya_base_model is not set in config.")

    return [
        sys.executable,
        str(tool_path),
        "--dataset-dir",
        str(dataset_dir),
        "--sd-scripts-dir",
        sd_scripts_dir,
        "--base-model",
        base_model,
        "--output-dir",
        output_dir,
        "--output-name",
        output_name,
        "--steps",
        str(steps),
        "--learning-rate",
        str(learning_rate),
        "--network-dim",
        str(network_dim),
        "--network-alpha",
        str(network_alpha),
        "--batch-size",
        str(batch_size),
        "--resolution",
        resolution,
        "--num-repeats",
        str(repeats),
    ]


def _run_command(command: list[str], dataset_dir: Path) -> dict:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "dataset_dir": str(dataset_dir),
    }
