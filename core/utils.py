from __future__ import annotations

import logging
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    array_bgr: np.ndarray


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    defaults = {
        "input_folder": "./generated",
        "output_folder": "./dnaduck_output",
        "database_path": "./dnaduck.sqlite3",
        "mode": "realism",
        "eps_realism": 0.31,
        "eps_anime": 0.47,
        "min_samples": 4,
        "assign_eps_realism": 0.27,
        "assign_eps_anime": 0.39,
        "model_name": "buffalo_l",
        "use_gpu": True,
        "det_size": [640, 640],
        "exclude_name_contains": ["_upscaled", ".thumb"],
        "overwrite_output": True,
        "identity_view_link_mode": "none",
        "lora_link_mode": "hardlink",
        "lora_min_images": 5,
        "lora_face_crop_enabled": False,
        "lora_face_crop_size": 1024,
        "lora_face_crop_padding": 0.35,
        "lora_face_crop_model_name": "buffalo_l",
        "lora_face_crop_use_gpu": False,
        "lora_trainer": "kohya_ss",
        "kohya_sd_scripts_dir": "",
        "kohya_base_model": "",
        "kohya_output_dir": "./dnaduck_output/trained_loras",
        "kohya_output_name": "dnaduck_lora",
        "kohya_train_steps": 1000,
        "kohya_learning_rate": 0.0001,
        "kohya_network_dim": 32,
        "kohya_network_alpha": 16,
        "kohya_batch_size": 1,
        "kohya_resolution": "1024,1024",
        "kohya_enable_bucket": True,
        "kohya_bucket_no_upscale": False,
        "kohya_min_bucket_reso": 256,
        "kohya_max_bucket_reso": 1536,
        "kohya_num_repeats": 10,
        "kohya_optimizer_type": "auto",
        "kohya_attention_backend": "auto",
        "kohya_max_data_loader_n_workers": 2,
        "kohya_persistent_data_loader_workers": True,
        "kohya_save_state": True,
        "kohya_save_state_every_n_steps": 250,
        "kohya_auto_resume": True,
        "kohya_resume_state": "",
        "lora_train_command": "",
        "auto_generate": {
            "enabled": False,
            "webbduck_url": "http://localhost:8020",
            "base_model": "",
            "scheduler": "UniPC",
            "steps": 30,
            "cfg": 7.5,
            "width": 1024,
            "height": 1024,
            "second_pass_model": "None",
            "negative_prompt": "low quality, blurry, bad anatomy, disfigured, extra limbs, bad hands",
            "target_count": 50,
            "max_attempts": 500,
            "assign_eps_realism": 0.27,
            "assign_eps_anime": 0.39,
            "prompt_templates": {
                "shot": {
                    "ultra closeup": 20,
                    "portrait": 30,
                    "3/4 shot": 10,
                    "full body": 10,
                    "side profile": 30,
                },
                "pose": {
                    "facing camera": 40,
                    "facing to the side": 25,
                    "standing": 15,
                    "sitting": 20,
                },
                "hair": {
                    "hair up": 50,
                    "hair down": 50,
                },
                "setting": {
                    "in a coffee shop": 15,
                    "in a forest": 15,
                    "hiking on a trail": 10,
                    "in a park": 15,
                    "on a city street": 15,
                    "in a studio": 10,
                    "in a garden": 10,
                    "at the beach": 10,
                },
                "clothing": {
                    "casual clothes": 30,
                    "formal wear": 15,
                    "summer dress": 20,
                    "leather jacket": 20,
                    "uniform": 15,
                },
                "style": {
                    "": 30,
                    "cinematic lighting": 25,
                    "soft natural lighting": 25,
                    "professional photography": 20,
                },
            },
        },
        "env": {},
        "log_level": "INFO",
    }
    merged = {**defaults, **raw}

    if str(merged["mode"]).lower() not in {"realism", "anime", "hybrid"}:
        raise ValueError("mode must be one of: realism, anime, hybrid")

    return merged


def resolve_runtime_paths(config: dict, config_path: Path) -> dict:
    base = config_path.parent.resolve()

    def _resolve(path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else (base / path).resolve()

    config = dict(config)
    config["input_folder"] = _resolve(str(config["input_folder"]))
    config["output_folder"] = _resolve(str(config["output_folder"]))
    config["database_path"] = _resolve(str(config["database_path"]))
    return config


def load_images(input_folder: Path) -> list[LoadedImage]:
    image_paths = discover_image_paths(input_folder)
    loaded, _ = load_images_from_paths(image_paths)
    return loaded


def discover_image_paths(input_folder: Path, exclude_name_contains: Iterable[str] | None = None) -> list[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    excluded_tokens = _normalize_excluded_tokens(exclude_name_contains)
    image_paths = sorted(
        p
        for p in input_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not _is_excluded_image_name(p, excluded_tokens)
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_folder}")
    return image_paths


def load_images_from_paths(
    paths: Iterable[Path],
    progress_callback=None,
) -> tuple[list[LoadedImage], list[Path]]:
    loaded: list[LoadedImage] = []
    unreadable: list[Path] = []
    ordered_paths = list(paths)
    total = len(ordered_paths)
    for index, path in enumerate(ordered_paths, start=1):
        array_bgr = cv2.imread(str(path))
        readable = array_bgr is not None
        if array_bgr is None:
            unreadable.append(path)
        else:
            loaded.append(LoadedImage(path=path, array_bgr=array_bgr))
        if callable(progress_callback):
            try:
                progress_callback(
                    {
                        "index": index,
                        "total": total,
                        "path": str(path),
                        "readable": readable,
                    }
                )
            except Exception:
                pass
    return loaded, unreadable


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_excluded_tokens(raw_tokens: Iterable[str] | None) -> list[str]:
    if raw_tokens is None:
        return []
    if isinstance(raw_tokens, str):
        values = [raw_tokens]
    else:
        values = list(raw_tokens)
    tokens: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        if token:
            tokens.append(token)
    return tokens


def _is_excluded_image_name(path: Path, excluded_tokens: list[str]) -> bool:
    if not excluded_tokens:
        return False
    name_lower = path.name.lower()
    return any(token in name_lower for token in excluded_tokens)


def write_clusters(
    labels: np.ndarray,
    embedded_paths: Iterable[Path],
    output_folder: Path,
    extra_unassigned: Iterable[Path] | None = None,
    overwrite_output: bool = True,
) -> dict:
    embedded_paths = list(embedded_paths)
    extra_unassigned = list(extra_unassigned or [])

    if len(labels) != len(embedded_paths):
        raise ValueError("labels and embedded_paths length mismatch")

    if overwrite_output and output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cluster_ids = sorted({int(v) for v in labels.tolist() if int(v) != -1})
    cluster_map = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}

    counts = {"clustered": 0, "noise": 0, "no_face": len(extra_unassigned), "clusters": len(cluster_ids)}

    for label, src_path in zip(labels.tolist(), embedded_paths):
        if int(label) == -1:
            target_dir = output_folder / "unassigned"
            counts["noise"] += 1
        else:
            target_dir = output_folder / f"identity_{cluster_map[int(label)]}"
            counts["clustered"] += 1
        _copy_into_dir(src_path, target_dir)

    for src_path in extra_unassigned:
        _copy_into_dir(src_path, output_folder / "unassigned")

    logging.getLogger(__name__).info(
        "Export complete: clusters=%d clustered=%d noise=%d no_face=%d output=%s",
        counts["clusters"],
        counts["clustered"],
        counts["noise"],
        counts["no_face"],
        output_folder,
    )
    return counts


def _copy_into_dir(src_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_path = target_dir / src_path.name
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
        return

    stem = src_path.stem
    suffix = src_path.suffix
    idx = 1
    while True:
        candidate = target_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src_path, candidate)
            return
        idx += 1


def ensure_clean_dir(path: Path, overwrite: bool = True) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def to_json_file(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def to_jsonl_file(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")
