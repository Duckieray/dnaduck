from __future__ import annotations
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .database import (
    compute_identity_drift,
    count_identity_images,
    count_unassigned_images,
    deserialize_embedding,
    fetch_image_by_path,
    list_all_images,
    list_identities,
    list_identity_images_page,
    list_no_face_images,
    list_noise_with_embeddings,
    list_unassigned_images,
    merge_identities,
    open_database,
    rebuild_identity_stats,
    serialize_embedding,
    update_image_embedding,
    update_image_status,
    update_identity_label,
)
from .pipeline import run_lora_export, run_scan, search_identity_candidates
from .utils import load_config, resolve_runtime_paths

_ACTIVE_TRAIN_PROCESS_LOCK = threading.Lock()
_ACTIVE_TRAIN_PROCESS: subprocess.Popen | None = None


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
        "caption_mode": str(result.get("caption_mode", "template_based")),
        "caption_template": str(result.get("caption_template", "{trigger}")),
        "image_preprocess": str(result.get("image_preprocess", "none")),
        "requested_identity_ids": list(result.get("requested_identity_ids", [])),
        "exported_identity_ids": list(result.get("exported_identity_ids", [])),
        "rich_captioning_status": "template_based",
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


def fetch_unassigned_images(
    config_path: Path,
    *,
    statuses: tuple[str, ...] | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        rows = list_unassigned_images(conn, statuses=statuses, limit=limit, offset=offset)
        total = count_unassigned_images(conn, statuses=statuses)
        images = [
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
        return {"images": images, "total": total, "limit": limit, "offset": offset}
    finally:
        conn.close()


def count_unassigned(config_path: Path) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        return {
            "needs_review": count_unassigned_images(conn, statuses=("noise", "no_face")),
            "noise": count_unassigned_images(conn, statuses=("noise",)),
            "no_face": count_unassigned_images(conn, statuses=("no_face",)),
        }
    finally:
        conn.close()


def reassign_image(config_path: Path, image_path: Path, identity_id: int) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    resolved = image_path.resolve()
    try:
        before = fetch_image_by_path(conn, resolved)
        if before is None:
            raise FileNotFoundError(f"Image is not tracked in DNADuck DB: {resolved}")
        ok = update_image_status(
            conn,
            path=resolved,
            status="assigned",
            identity_id=int(identity_id),
            clear_embedding=False,
        )
        if not ok:
            raise FileNotFoundError(f"Image is not tracked in DNADuck DB: {resolved}")
        rebuild_identity_stats(conn)
        after = fetch_image_by_path(conn, resolved)
        return {
            "ok": True,
            "image_path": str(resolved),
            "identity_id": int(identity_id),
            "before_status": None if before is None else str(before["status"]),
            "before_identity_id": None if before is None else before["identity_id"],
            "after_status": None if after is None else str(after["status"]),
            "after_identity_id": None if after is None else after["identity_id"],
        }
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


def _favorites_path(config_path: Path) -> Path:
    return config_path.resolve().parent / ".dnaduck_favorites.json"


def get_favorites(config_path: Path) -> set[str]:
    sp = _favorites_path(config_path)
    if not sp.exists():
        return set()
    try:
        data = json.loads(sp.read_text())
        return set(data.get("favorites", []))
    except Exception:
        return set()


def set_image_favorite(config_path: Path, image_path: str, favorite: bool) -> dict:
    sp = _favorites_path(config_path)
    favs = get_favorites(config_path)
    resolved = str(Path(image_path).resolve())
    if favorite:
        favs.add(resolved)
    else:
        favs.discard(resolved)
    sp.write_text(json.dumps({"favorites": sorted(favs)}, indent=2))
    return {"ok": True, "image_path": resolved, "favorite": favorite}


def recluster_noise(config_path: Path) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        rows = list_noise_with_embeddings(conn)
    finally:
        conn.close()

    if not rows:
        return {"clusters": [], "images": [], "total": 0}

    import numpy as np

    paths = []
    vectors = []
    for row in rows:
        embedding = deserialize_embedding(row["embedding"], row["embedding_dim"])
        if embedding is not None:
            paths.append(str(row["path"]))
            vectors.append(embedding)

    if len(vectors) < 2:
        return {"clusters": [], "images": [{"path": p, "status": "noise"} for p in paths], "total": len(paths)}

    matrix = np.stack(vectors, axis=0)

    from sklearn.cluster import DBSCAN

    eps = float(config.get("eps_realism", 0.31))
    min_samples = int(config.get("min_samples", 4))
    mode = str(config.get("mode", "realism")).strip().lower()
    if mode == "anime":
        eps = float(config.get("eps_anime", 0.47))
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(matrix)

    clusters: dict[int, list[str]] = {}
    standalone: list[str] = []
    for path, label in zip(paths, labels.tolist()):
        if int(label) == -1:
            standalone.append(path)
        else:
            clusters.setdefault(int(label), []).append(path)

    cluster_list = [
        {"cluster_id": cid, "images": [{"path": p, "status": "noise", "cluster_id": cid} for p in members]}
        for cid, members in sorted(clusters.items())
    ]

    all_images: list[dict] = []
    for c in cluster_list:
        all_images.extend(c["images"])
    for p in standalone:
        all_images.append({"path": p, "status": "noise"})

    return {
        "clusters": cluster_list,
        "images": all_images,
        "total": len(all_images),
        "recluster_eps": eps,
    }


def reanalyze_no_face(config_path: Path) -> dict:
    config = load_runtime_config(config_path=config_path)
    conn = open_database(Path(config["database_path"]))
    try:
        rows = list_no_face_images(conn)
    finally:
        conn.close()

    if not rows:
        return {"clusters": [], "images": [], "total": 0, "reanalyzed": 0, "new_faces": 0}

    from .embedder import FaceEmbedder

    embedder = FaceEmbedder(config)
    import numpy as np
    from pathlib import Path

    reanalyzed = 0
    new_faces = 0
    new_embeddings: list[tuple[str, np.ndarray]] = []
    still_no_face: list[str] = []

    for row in rows:
        image_path = Path(str(row["path"]))
        if not image_path.exists():
            still_no_face.append(str(image_path))
            continue

        reanalyzed += 1
        import cv2

        array_bgr = cv2.imread(str(image_path))
        if array_bgr is None:
            still_no_face.append(str(image_path))
            continue

        from .embedder import LoadedImage

        loaded = LoadedImage(path=image_path, array_bgr=array_bgr)
        result = embedder.extract([loaded])

        if str(image_path) in [str(p) for p in result.embedded_paths]:
            idx = [str(p) for p in result.embedded_paths].index(str(image_path))
            new_embeddings.append((str(image_path), result.embeddings[idx]))
            new_faces += 1
        else:
            still_no_face.append(str(image_path))

    conn2 = open_database(Path(config["database_path"]))
    try:
        from .database import serialize_embedding, update_image_embedding

        for path_str, vector in new_embeddings:
            blob, dim = serialize_embedding(vector)
            update_image_embedding(
                conn2,
                path=Path(path_str),
                status="noise",
                identity_id=None,
                embedding=blob,
                embedding_dim=dim,
            )
        conn2.commit()
    finally:
        conn2.close()

    result_images = (
        [{"path": p, "status": "noise"} for p, _ in new_embeddings]
        + [{"path": p, "status": "no_face"} for p in still_no_face]
    )

    return {
        "images": result_images,
        "total": len(result_images),
        "reanalyzed": reanalyzed,
        "new_faces": new_faces,
    }
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


def trigger_lora_training(
    config_path: Path,
    overrides: dict | None = None,
    progress_callback=None,
    stop_callback=None,
) -> dict:
    config = load_runtime_config(config_path=config_path, overrides=overrides)
    dataset_dir = Path(config["output_folder"]).resolve() / "lora_export"

    # Build extra environment from config
    raw_env = config.get("env", {})
    env_overrides: dict[str, str] | None = None
    if isinstance(raw_env, dict):
        env_overrides = {}
        for k, v in raw_env.items():
            key = str(k).strip()
            val = str(v).strip() if v is not None else ""
            if key and val:
                env_overrides[key] = val
        if not env_overrides:
            env_overrides = None

    command_template = str(config.get("lora_train_command", "")).strip()
    trainer = str(config.get("lora_trainer", "kohya_ss")).strip().lower()

    if command_template:
        command = shlex.split(command_template.format(dataset_dir=str(dataset_dir)))
        return _run_command(
            command,
            dataset_dir=dataset_dir,
            env_overrides=env_overrides,
            progress_callback=progress_callback,
            stop_callback=stop_callback,
        )

    if trainer == "kohya_ss":
        command, output_dir, output_name = _build_kohya_command(config, dataset_dir=dataset_dir)
        return _run_command(
            command,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_name_prefix=output_name,
            env_overrides=env_overrides,
            progress_callback=progress_callback,
            stop_callback=stop_callback,
        )

    raise ValueError(
        "No supported training configuration found. "
        "Set lora_train_command or lora_trainer='kohya_ss' in config."
    )


def request_active_training_stop() -> bool:
    proc: subprocess.Popen | None
    with _ACTIVE_TRAIN_PROCESS_LOCK:
        proc = _ACTIVE_TRAIN_PROCESS
    if proc is None:
        return False
    if proc.poll() is not None:
        return False
    _terminate_process_tree(proc, graceful_timeout_s=8.0)
    return True


def is_active_training_running() -> bool:
    with _ACTIVE_TRAIN_PROCESS_LOCK:
        proc = _ACTIVE_TRAIN_PROCESS
        return bool(proc is not None and proc.poll() is None)


def _build_kohya_command(config: dict, dataset_dir: Path) -> tuple[list[str], Path, str]:
    dnaduck_root = Path(__file__).resolve().parent.parent
    tool_path = dnaduck_root / "tools" / "train_kohya_lora.py"
    sd_scripts_dir_raw = str(config.get("kohya_sd_scripts_dir", "")).strip()
    base_model_raw = str(config.get("kohya_base_model", "")).strip()
    output_dir_raw = str(config.get("kohya_output_dir", "")).strip() or str(
        (Path(config["output_folder"]).resolve() / "trained_loras")
    )
    output_name = str(config.get("kohya_output_name", "dnaduck_lora")).strip()
    steps = int(config.get("kohya_train_steps", 1000))
    learning_rate = float(config.get("kohya_learning_rate", 1e-4))
    network_dim = int(config.get("kohya_network_dim", 32))
    network_alpha = int(config.get("kohya_network_alpha", 16))
    batch_size = int(config.get("kohya_batch_size", 1))
    resolution = str(config.get("kohya_resolution", "1024,1024"))
    enable_bucket = _as_bool(config.get("kohya_enable_bucket", True), default=True)
    bucket_no_upscale = _as_bool(config.get("kohya_bucket_no_upscale", False), default=False)
    min_bucket_reso = int(config.get("kohya_min_bucket_reso", 256))
    max_bucket_reso = int(config.get("kohya_max_bucket_reso", 1536))
    repeats = int(config.get("kohya_num_repeats", 10))
    optimizer_type = str(config.get("kohya_optimizer_type", "auto")).strip() or "auto"
    attention_backend = str(config.get("kohya_attention_backend", "auto")).strip() or "auto"
    max_data_loader_n_workers = int(config.get("kohya_max_data_loader_n_workers", 2))
    persistent_data_loader_workers = _as_bool(
        config.get("kohya_persistent_data_loader_workers", True),
        default=True,
    )
    save_state = _as_bool(config.get("kohya_save_state", True), default=True)
    save_state_every_n_steps = int(config.get("kohya_save_state_every_n_steps", 100))
    auto_resume = _as_bool(config.get("kohya_auto_resume", True), default=True)
    resume_state_raw = str(config.get("kohya_resume_state", "")).strip()

    optimizer_normalized = optimizer_type.lower()
    if optimizer_normalized not in {"auto", "adamw", "adamw8bit"}:
        raise ValueError(
            "kohya_optimizer_type must be one of: auto, AdamW, AdamW8bit."
        )
    attention_normalized = attention_backend.lower()
    if attention_normalized not in {"auto", "xformers", "sdpa", "none", "off", "disabled"}:
        raise ValueError(
            "kohya_attention_backend must be one of: auto, xformers, sdpa, none."
        )

    if not sd_scripts_dir_raw:
        raise ValueError("kohya_sd_scripts_dir is not set in config.")
    if not base_model_raw:
        raise ValueError("kohya_base_model is not set in config.")

    missing_modules = [
        module_name
        for module_name in ("toml", "accelerate")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_modules:
        missing_text = ", ".join(sorted(missing_modules))
        raise ValueError(
            "Missing trainer dependencies: "
            f"{missing_text}. Install them in the active DNADuck Python env "
            "(example: `pip install toml accelerate`)."
        )
    sd_scripts_dir_path = Path(sd_scripts_dir_raw)
    if not sd_scripts_dir_path.is_absolute():
        sd_scripts_dir_path = (dnaduck_root / sd_scripts_dir_path).resolve()
    else:
        sd_scripts_dir_path = sd_scripts_dir_path.resolve()
    if not (sd_scripts_dir_path / "sdxl_train_network.py").exists():
        raise ValueError(
            f"kohya_sd_scripts_dir is invalid: {sd_scripts_dir_path} "
            "(missing sdxl_train_network.py)."
        )

    base_model_lower = base_model_raw.strip().lower()
    if "your_sdxl_model" in base_model_lower:
        raise ValueError(
            "kohya_base_model is still a placeholder. "
            "Set it to your real model file path in config.yaml."
        )
    base_model = base_model_raw
    looks_like_local_path = (
        base_model_raw.startswith(".")
        or base_model_raw.startswith("/")
        or base_model_raw.startswith("\\")
        or (len(base_model_raw) >= 2 and base_model_raw[1] == ":")
        or bool(Path(base_model_raw).suffix)
    )
    if looks_like_local_path:
        base_model_path = Path(base_model_raw)
        if not base_model_path.is_absolute():
            base_model_path = (dnaduck_root / base_model_path).resolve()
        else:
            base_model_path = base_model_path.resolve()
        if not base_model_path.exists():
            raise ValueError(
                "kohya_base_model does not exist: "
                f"{base_model_path}. Update config.yaml."
            )
        base_model = str(base_model_path)

    output_dir_path = Path(output_dir_raw)
    if output_dir_path.is_absolute():
        output_dir = str(output_dir_path.resolve())
    else:
        output_dir = str((dnaduck_root / output_dir_path).resolve())

    resume_state: str | None = None
    if resume_state_raw:
        resume_state_path = Path(resume_state_raw)
        if not resume_state_path.is_absolute():
            resume_state_path = (dnaduck_root / resume_state_path).resolve()
        else:
            resume_state_path = resume_state_path.resolve()
        if not resume_state_path.exists() or not resume_state_path.is_dir():
            raise ValueError(
                "kohya_resume_state does not exist or is not a directory: "
                f"{resume_state_path}"
            )
        resume_state = str(resume_state_path)

    command = [
            sys.executable,
            str(tool_path),
            "--dataset-dir",
            str(dataset_dir),
            "--sd-scripts-dir",
            str(sd_scripts_dir_path),
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
            "--enable-bucket" if enable_bucket else "--disable-bucket",
            "--bucket-no-upscale" if bucket_no_upscale else "",
            "--min-bucket-reso",
            str(max(64, min_bucket_reso)),
            "--max-bucket-reso",
            str(max(64, max_bucket_reso)),
            "--num-repeats",
            str(repeats),
            "--optimizer-type",
            optimizer_type,
            "--attention",
            attention_backend,
            "--max-data-loader-workers",
            str(max(0, max_data_loader_n_workers)),
            "--persistent-data-loader-workers" if persistent_data_loader_workers else "--no-persistent-data-loader-workers",
            "--save-state" if save_state else "--no-save-state",
            "--save-state-every-steps",
            str(max(1, save_state_every_n_steps)),
            "--auto-resume" if auto_resume else "--no-auto-resume",
    ]
    if resume_state:
        command.extend(["--resume-state", resume_state])
    command = [token for token in command if token]

    return (
        command,
        Path(output_dir).resolve(),
        output_name,
    )


def _run_command(
    command: list[str],
    dataset_dir: Path,
    *,
    output_dir: Path | None = None,
    output_name_prefix: str | None = None,
    env_overrides: dict[str, str] | None = None,
    progress_callback=None,
    stop_callback=None,
) -> dict:
    global _ACTIVE_TRAIN_PROCESS
    before_files: list[Path] = []
    after_files: list[Path] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        before_files = sorted(output_dir.glob("*.safetensors"))

    expected_steps = _extract_int_arg(command, "--steps")
    tracker = _TrainProgressTracker(
        progress_callback=progress_callback,
        expected_steps=expected_steps,
    )
    tracker.emit({"stage": "training_starting", "message": "Launching trainer..."})

    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)

    popen_kwargs: dict = {
        "args": command,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        popen_kwargs["preexec_fn"] = os.setsid
    proc = subprocess.Popen(**popen_kwargs)
    with _ACTIVE_TRAIN_PROCESS_LOCK:
        _ACTIVE_TRAIN_PROCESS = proc
    try:
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stdout_thread = threading.Thread(
            target=_consume_process_stream,
            args=(proc.stdout, stdout_chunks, tracker.feed),
            daemon=True,
            name="dnaduck-train-stdout",
        )
        stderr_thread = threading.Thread(
            target=_consume_process_stream,
            args=(proc.stderr, stderr_chunks, tracker.feed),
            daemon=True,
            name="dnaduck-train-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()

        stopped_by_user = False
        while True:
            polled = proc.poll()
            if polled is not None:
                returncode = int(polled)
                break
            should_stop = False
            if callable(stop_callback):
                try:
                    should_stop = bool(stop_callback())
                except Exception:
                    should_stop = False
            if should_stop:
                stopped_by_user = True
                tracker.emit({"stage": "training_pausing", "message": "Pause requested. Stopping trainer..."})
                _terminate_process_tree(proc, graceful_timeout_s=10.0)
                polled = _wait_for_process_exit(proc, timeout_s=20.0)
                if polled is None:
                    _terminate_process_tree(proc, graceful_timeout_s=2.0)
                    polled = _wait_for_process_exit(proc, timeout_s=5.0)
                returncode = int(polled) if polled is not None else -9
                break
            time.sleep(0.25)
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)
        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        if stopped_by_user:
            tracker.emit({"stage": "training_paused", "message": "Training paused by user."})
        elif returncode == 0:
            tracker.emit({"stage": "training_complete", "message": "Training command completed."})
        else:
            tracker.emit({"stage": "training_failed", "message": f"Training command failed (code {returncode})."})

        if output_dir is not None:
            after_files = sorted(output_dir.glob("*.safetensors"))

        before_set = {str(path.resolve()) for path in before_files}
        new_artifacts = [
            str(path.resolve())
            for path in after_files
            if str(path.resolve()) not in before_set
        ]
        log_file = _write_train_log(
            dataset_dir=dataset_dir,
            command=command,
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            output_dir=output_dir,
            output_name_prefix=output_name_prefix,
            artifacts_before=len(before_files),
            artifacts_after=len(after_files),
            new_artifacts=new_artifacts,
        )
        return {
            "command": command,
            "returncode": returncode,
            "stdout": stdout_text[-4000:],
            "stderr": stderr_text[-4000:],
            "dataset_dir": str(dataset_dir),
            "output_dir": None if output_dir is None else str(output_dir.resolve()),
            "output_name_prefix": output_name_prefix,
            "artifacts_before": len(before_files),
            "artifacts_after": len(after_files),
            "new_artifacts": new_artifacts,
            "log_file": str(log_file),
            "stopped_by_user": bool(stopped_by_user),
        }
    finally:
        with _ACTIVE_TRAIN_PROCESS_LOCK:
            if _ACTIVE_TRAIN_PROCESS is proc:
                _ACTIVE_TRAIN_PROCESS = None


def _terminate_process_tree(proc: subprocess.Popen, *, graceful_timeout_s: float) -> None:
    pid = int(proc.pid)
    if os.name == "nt":
        # Kill the full tree immediately on Windows. If the parent exits first,
        # children can outlive it and keep GPU work running invisibly.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        deadline = time.monotonic() + max(0.5, float(graceful_timeout_s))
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.25)
        try:
            proc.kill()
        except Exception:
            pass
        return

    if proc.poll() is not None:
        return

    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    else:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.monotonic() + max(0.5, float(graceful_timeout_s))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.25)

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    else:
        try:
            proc.kill()
        except Exception:
            pass


def _wait_for_process_exit(proc: subprocess.Popen, *, timeout_s: float) -> int | None:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        polled = proc.poll()
        if polled is not None:
            return int(polled)
        time.sleep(0.2)
    polled = proc.poll()
    return None if polled is None else int(polled)


def _as_bool(value, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


_PROGRESS_COUNTER_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


def _extract_int_arg(command: list[str], key: str) -> int | None:
    for index, token in enumerate(command):
        if token == key and index + 1 < len(command):
            try:
                return int(str(command[index + 1]).strip())
            except Exception:
                return None
    return None


def _consume_process_stream(stream, sink_chunks: list[str], on_fragment) -> None:
    if stream is None:
        return
    fragment_chars: list[str] = []
    try:
        while True:
            ch = stream.read(1)
            if ch == "":
                break
            sink_chunks.append(ch)
            if ch in {"\n", "\r"}:
                fragment = "".join(fragment_chars).strip()
                fragment_chars.clear()
                if fragment:
                    on_fragment(fragment)
            else:
                fragment_chars.append(ch)
    finally:
        fragment = "".join(fragment_chars).strip()
        if fragment:
            on_fragment(fragment)
        try:
            stream.close()
        except Exception:
            pass


class _TrainProgressTracker:
    def __init__(self, progress_callback, expected_steps: int | None):
        self._progress_callback = progress_callback
        self._expected_steps = int(expected_steps) if isinstance(expected_steps, int) and expected_steps > 0 else None
        self._lock = threading.Lock()
        self._training_started = False
        self._training_started_at_monotonic: float | None = None
        self._last_step = -1
        self._last_total = -1
        self._last_emit_at_monotonic = 0.0
        self._last_stage = ""
        self._step_samples: list[tuple[float, int]] = []
        self._last_rate_steps_per_s: float | None = None

    def emit(self, payload: dict) -> None:
        if not callable(self._progress_callback):
            return
        try:
            self._progress_callback(payload)
        except Exception:
            return

    def feed(self, text: str) -> None:
        clean = str(text or "").strip()
        if not clean:
            return
        lower = clean.lower()
        with self._lock:
            if "caching latents" in lower:
                self._emit_stage_once("training_preparing", "Caching training latents...")
            elif "prepare optimizer" in lower or "preparing accelerator" in lower:
                self._emit_stage_once("training_preparing", "Preparing optimizer and dataloader...")

            if "running training" in lower or "学習開始" in clean:
                if not self._training_started:
                    self._training_started = True
                    self._training_started_at_monotonic = time.monotonic()
                    # Reset prep counters as we transition to optimizer steps.
                    self._last_step = -1
                    self._last_total = -1
                    self._step_samples = []
                    self._last_rate_steps_per_s = None
                    self._emit_training_started()

            # First optimizer step can take a long time while GPU kernels warm up.
            # Emit Step 0/total heartbeat so UI doesn't look frozen before first tqdm counter.
            if self._training_started and self._expected_steps and self._last_step < 0:
                now = time.monotonic()
                if (now - self._last_emit_at_monotonic) >= 4.0:
                    self._last_emit_at_monotonic = now
                    self.emit(
                        {
                            "stage": "training",
                            "processed_count": 0,
                            "total_count": int(self._expected_steps),
                            "progress_pct": 0.0,
                            "message": f"Training in progress... Step 0/{int(self._expected_steps)}",
                        }
                    )

            step, total = self._extract_counter(clean)
            if step is None or total is None or total <= 0 or step < 0:
                return

            if not self._should_accept_counter(step=step, total=total):
                return

            now = time.monotonic()
            if step < self._last_step:
                return
            if step == self._last_step and total == self._last_total and (now - self._last_emit_at_monotonic) < 2.0:
                return

            if (
                not self._training_started
                and self._last_stage != "training_preparing"
                and self._looks_like_training_counter(total=total)
            ):
                self._training_started = True
                self._training_started_at_monotonic = now
                self._last_step = -1
                self._last_total = -1
                self._step_samples = []
                self._last_rate_steps_per_s = None
                self._emit_training_started()

            payload: dict[str, object] = {
                "stage": "training" if self._training_started else "training_preparing",
            }
            if self._training_started:
                payload["processed_count"] = int(step)
                payload["total_count"] = int(total)
                payload["message"] = f"Training in progress... Step {step}/{total}"
                if total >= step:
                    payload["progress_pct"] = float(step) / float(total) * 100.0

                rate = self._update_and_estimate_rate(now=now, step=int(step))
                started_at = self._training_started_at_monotonic or now
                elapsed = max(0.0, now - started_at)
                min_steps_for_eta = max(10, int(total * 0.02))
                if (
                    rate is not None
                    and rate > 0
                    and step >= min_steps_for_eta
                    and elapsed >= 20.0
                    and total >= step
                ):
                    remaining = max(0.0, float(total) - float(step))
                    eta_raw = remaining / rate
                    # Avoid displaying impossible ETA=0 before completion.
                    if step >= total:
                        payload["eta_s"] = 0
                    elif eta_raw >= 1.0:
                        payload["eta_s"] = int(round(eta_raw))
            else:
                payload["processed_count"] = int(step)
                payload["total_count"] = int(total)
                if total >= step:
                    payload["progress_pct"] = float(step) / float(total) * 100.0
                payload["message"] = f"Preparing training data... {int(step)}/{int(total)}"

            self._last_step = int(step)
            self._last_total = int(total)
            self._last_emit_at_monotonic = now
            self.emit(payload)

    def _emit_stage_once(self, stage: str, message: str) -> None:
        if self._last_stage == stage:
            return
        self._last_stage = stage
        self.emit({"stage": stage, "message": message})

    def _emit_training_started(self) -> None:
        self._last_stage = "training"
        payload: dict[str, object] = {"stage": "training", "message": "Training in progress..."}
        if self._expected_steps and self._expected_steps > 0:
            payload["processed_count"] = 0
            payload["total_count"] = int(self._expected_steps)
            payload["progress_pct"] = 0.0
            payload["eta_s"] = None
            payload["message"] = f"Training in progress... Step 0/{int(self._expected_steps)}"
        self.emit(payload)

    def _extract_counter(self, text: str) -> tuple[int | None, int | None]:
        matches = list(_PROGRESS_COUNTER_RE.finditer(text))
        if not matches:
            return None, None
        last = matches[-1]
        try:
            return int(last.group(1)), int(last.group(2))
        except Exception:
            return None, None

    def _should_accept_counter(self, *, step: int, total: int) -> bool:
        if total <= 0:
            return False
        if self._expected_steps is not None:
            tolerance = max(2, int(self._expected_steps * 0.05))
            matches_expected = abs(total - self._expected_steps) <= tolerance
            if matches_expected:
                return True
            if self._training_started:
                # During step training, ignore unrelated counters such as cached-image totals.
                return False
            # During latent caching and preparation, counters may use dataset-sized totals.
            return self._last_stage == "training_preparing" and step <= total
        if not self._training_started:
            return self._last_stage == "training_preparing" and step <= total
        return step <= total

    def _looks_like_training_counter(self, *, total: int) -> bool:
        if self._expected_steps is None:
            return False
        tolerance = max(2, int(self._expected_steps * 0.05))
        return abs(int(total) - int(self._expected_steps)) <= tolerance

    def _update_and_estimate_rate(self, *, now: float, step: int) -> float | None:
        if step < 0:
            return self._last_rate_steps_per_s

        if not self._step_samples or step > self._step_samples[-1][1]:
            self._step_samples.append((now, step))

        window_s = 180.0
        cutoff = now - window_s
        self._step_samples = [row for row in self._step_samples if row[0] >= cutoff]
        if len(self._step_samples) < 2:
            return self._last_rate_steps_per_s

        first_t, first_step = self._step_samples[0]
        last_t, last_step = self._step_samples[-1]
        delta_steps = max(0, last_step - first_step)
        delta_t = max(0.001, last_t - first_t)
        # If buffered output flushes many steps at once, this window is unreliable.
        # For very slow runs (~1 step every 1-2 minutes), allow ETA after 2 observed steps.
        if delta_steps <= 0 or delta_t < 20.0 or delta_steps < 2:
            return self._last_rate_steps_per_s

        instant = float(delta_steps) / float(delta_t)
        prev = self._last_rate_steps_per_s
        if prev is None:
            self._last_rate_steps_per_s = instant
        else:
            # Smooth short spikes to prevent wild ETA jumps.
            self._last_rate_steps_per_s = (prev * 0.70) + (instant * 0.30)
        return self._last_rate_steps_per_s


def auto_generate(
    config_path: Path,
    identity_id: int,
    target_count: int = 50,
    max_attempts: int = 500,
    assign_eps_realism: float | None = None,
    assign_eps_anime: float | None = None,
    target_identity_id: int | None = None,
    new_character_label: str | None = None,
) -> dict:
    """Start auto-generation in a background thread. Returns status immediately."""
    import numpy as np
    from .autogen import init_auto_generate_status, run_auto_generate
    from .database import create_identity, open_database
    from .utils import load_config, get_logger

    _log = get_logger("dnaduck.service")
    _log.warning(
        "auto_generate called: identity_id=%s target_identity_id=%s new_character_label=%s",
        identity_id, target_identity_id, new_character_label,
    )

    resolved_target: int | None = None
    if new_character_label:
        cfg = load_config(config_path.resolve())
        db_path = Path(cfg["database_path"])
        conn = open_database(db_path)
        try:
            dummy = np.zeros(768, dtype=np.float32)
            new_id = create_identity(conn, centroid=dummy, label=new_character_label.strip())
            conn.commit()
            resolved_target = new_id
        finally:
            conn.close()
    elif target_identity_id:
        resolved_target = int(target_identity_id)

    # Initialize status BEFORE starting the thread so the first
    # get_auto_generate_status() call sees running=True.
    init_auto_generate_status(
        identity_id=int(identity_id),
        target_count=int(target_count),
        max_attempts=int(max_attempts),
    )

    thread = threading.Thread(
        target=run_auto_generate,
        kwargs={
            "config_path": config_path,
            "identity_id": int(identity_id),
            "target_count": int(target_count),
            "max_attempts": int(max_attempts),
            "assign_eps_realism": assign_eps_realism,
            "assign_eps_anime": assign_eps_anime,
            "target_identity_id": resolved_target,
        },
        daemon=True,
        name=f"dnaduck-autogen-{int(identity_id)}",
    )
    _log.warning(
        "Starting autogen thread: identity_id=%s target_identity_id=%s resolved_target=%s",
        identity_id, target_identity_id, resolved_target,
    )
    thread.start()
    return get_auto_generate_status()


def cancel_auto_generate() -> dict:
    from .autogen import cancel_auto_generate as _cancel
    return _cancel()


def get_auto_generate_status() -> dict:
    from .autogen import get_auto_generate_status as _status
    return _status()


def _write_train_log(
    *,
    dataset_dir: Path,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    output_dir: Path | None,
    output_name_prefix: str | None,
    artifacts_before: int,
    artifacts_after: int,
    new_artifacts: list[str],
) -> Path:
    output_root = dataset_dir.parent
    logs_dir = output_root / "train_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = logs_dir / f"train_{stamp}.log"

    lines: list[str] = []
    lines.append(f"timestamp_utc: {stamp}")
    lines.append(f"dataset_dir: {dataset_dir}")
    lines.append(f"output_dir: {output_dir if output_dir is not None else ''}")
    lines.append(f"output_name_prefix: {output_name_prefix or ''}")
    lines.append(f"returncode: {returncode}")
    lines.append(f"artifacts_before: {artifacts_before}")
    lines.append(f"artifacts_after: {artifacts_after}")
    lines.append(f"new_artifacts_count: {len(new_artifacts)}")
    for path in new_artifacts:
        lines.append(f"new_artifact: {path}")
    lines.append("")
    lines.append("command:")
    lines.append(" ".join(command))
    lines.append("")
    lines.append("stdout:")
    lines.append(stdout or "")
    lines.append("")
    lines.append("stderr:")
    lines.append(stderr or "")
    log_file.write_text("\n".join(lines), encoding="utf-8")
    return log_file
