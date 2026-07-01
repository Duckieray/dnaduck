from __future__ import annotations

import logging
import random
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .embedder import FaceEmbedder, LoadedImage

log = logging.getLogger("dnaduck.autogen")

_AUTOGEN_CANCEL: threading.Event | None = None
_AUTOGEN_LOCK = threading.Lock()
_AUTOGEN_STATUS: dict = {}


def _resolve_weighted_list(raw) -> list[tuple[str, float]]:
    """Convert a template category to a list of (text, weight) tuples.

    Supports three formats:
      - list of strings: uniform weight (1.0 each)
      - dict of str->number: weighted by value
      - list of dicts with "text"/"weight" keys: explicit weighted
    """
    if isinstance(raw, list):
        items: list[tuple[str, float]] = []
        for entry in raw:
            if isinstance(entry, str):
                items.append((entry, 1.0))
            elif isinstance(entry, dict):
                text = str(entry.get("text", ""))
                w = float(entry.get("weight", 1.0))
                if text:
                    items.append((text, max(0.0, w)))
        return items
    if isinstance(raw, dict):
        items = [(str(k), max(0.0, float(v))) for k, v in raw.items()]
        # Filter out empty-text entries
        return [(t, w) for t, w in items if t]
    return []


def _weighted_choice(items: list[tuple[str, float]]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0][0]
    texts, weights = zip(*items)
    if sum(weights) <= 0:
        return random.choice(texts)
    return random.choices(texts, weights=weights, k=1)[0]


def _load_prompt_templates(config: dict) -> dict[str, list[tuple[str, float]]]:
    raw = config.get("auto_generate", {})
    templates = raw.get("prompt_templates", {})
    return {
        "shot": _resolve_weighted_list(templates.get("shot", [
            "ultra closeup",
            "portrait",
            "3/4 shot",
            "full body",
            "side profile",
        ])),
        "pose": _resolve_weighted_list(templates.get("pose", [
            "facing camera",
            "facing to the side",
            "standing",
            "sitting",
        ])),
        "hair": _resolve_weighted_list(templates.get("hair", [
            "hair up",
            "hair down",
        ])),
        "setting": _resolve_weighted_list(templates.get("setting", [
            "in a coffee shop",
            "in a forest",
            "hiking on a trail",
            "in a park",
            "on a city street",
            "in a studio",
            "in a garden",
            "at the beach",
        ])),
        "clothing": _resolve_weighted_list(templates.get("clothing", [
            "casual clothes",
            "formal wear",
            "summer dress",
            "leather jacket",
            "uniform",
        ])),
        "style": _resolve_weighted_list(templates.get("style", [
            "",
            "cinematic lighting",
            "soft natural lighting",
            "professional photography",
        ])),
    }


def _build_prompt(character_label: str, templates: dict[str, list[tuple[str, float]]]) -> str:
    shot = _weighted_choice(templates["shot"])
    pose = _weighted_choice(templates["pose"])
    hair = _weighted_choice(templates["hair"])
    setting = _weighted_choice(templates["setting"])
    clothing = _weighted_choice(templates["clothing"])
    style = _weighted_choice(templates["style"])
    parts = [f"{character_label}, {shot}, {clothing}, {hair}, {pose}, {setting}"]
    if style:
        parts.append(style)
    parts.append("detailed face, high quality")
    return ", ".join(parts)


def _generate_image_via_webbduck(
    prompt: str,
    negative_prompt: str,
    webbduck_url: str,
    base_model: str,
    cfg: float,
    steps: int,
    width: int,
    height: int,
    scheduler: str,
    second_pass_model: str,
) -> str | None:
    url = f"{webbduck_url.rstrip('/')}/test"
    data = urllib.parse.urlencode({
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "base_model": base_model,
        "steps": str(steps),
        "cfg": str(cfg),
        "width": str(width),
        "height": str(height),
        "scheduler": scheduler,
        "second_pass_model": second_pass_model or "None",
        "second_pass_mode": "auto",
        "wait_for_result": "True",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            import json
            body = json.loads(resp.read().decode())
            if "images" in body and len(body["images"]) > 0:
                return str(body["images"][0])
            if "error" in body:
                log.error("Generation error: %s", body["error"])
            return None
    except Exception as exc:
        log.error("WebbDuck generation request failed: %s", exc)
        return None


def _match_to_identity(
    image_path: Path,
    embedder: FaceEmbedder,
    identity_centroid: np.ndarray,
    assign_eps: float,
) -> bool:
    array_bgr = cv2.imread(str(image_path))
    if array_bgr is None:
        return False
    loaded = LoadedImage(path=image_path, array_bgr=array_bgr)
    result = embedder.extract([loaded])
    if str(image_path) not in result.embedded_paths:
        return False
    idx = result.embedded_paths.index(str(image_path))
    vector = result.embeddings[idx]
    dist = float(np.dot(vector, identity_centroid))
    cos_sim = float(np.clip(dist, -1.0, 1.0))
    cos_dist = 1.0 - cos_sim
    return cos_dist <= assign_eps


def _add_to_dataset(config: dict, image_path: Path, identity_id: int) -> None:
    import hashlib
    import os

    from .database import (
        open_database,
        serialize_embedding,
        update_image_embedding,
        update_image_status,
    )

    db_path = Path(config["database_path"])
    conn = open_database(db_path)
    try:
        stat = image_path.stat()
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        row = conn.execute(
            """
            SELECT id FROM images WHERE path = ?
            """,
            (str(image_path),),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO images (path, sha256, size_bytes, mtime, status, identity_id, created_at, updated_at, last_seen_at)
                VALUES (?, ?, ?, ?, 'assigned', ?, datetime('now'), datetime('now'), datetime('now'))
                """,
                (str(image_path), sha, stat.st_size, stat.st_mtime, int(identity_id)),
            )
        else:
            update_image_status(
                conn, path=image_path, status="assigned", identity_id=int(identity_id), clear_embedding=False,
            )
        conn.commit()
    finally:
        conn.close()


def _get_identity_centroid(config: dict, identity_id: int) -> np.ndarray | None:
    from .database import deserialize_embedding, open_database

    db_path = Path(config["database_path"])
    conn = open_database(db_path)
    try:
        row = conn.execute(
            "SELECT centroid, centroid_dim FROM identities WHERE id = ?",
            (int(identity_id),),
        ).fetchone()
        if row is None:
            return None
        return deserialize_embedding(row["centroid"], row["centroid_dim"])
    finally:
        conn.close()


def _get_identity_label(config: dict, identity_id: int) -> str | None:
    from .database import open_database

    db_path = Path(config["database_path"])
    conn = open_database(db_path)
    try:
        row = conn.execute(
            "SELECT label FROM identities WHERE id = ?",
            (int(identity_id),),
        ).fetchone()
        return str(row["label"]) if row and row["label"] else None
    finally:
        conn.close()


def run_auto_generate(
    config_path: Path,
    identity_id: int,
    target_count: int = 50,
    max_attempts: int = 500,
    progress_callback=None,
) -> dict:
    global _AUTOGEN_CANCEL, _AUTOGEN_STATUS

    from .utils import load_config, resolve_runtime_paths

    config = load_config(config_path.resolve())
    config = resolve_runtime_paths(config, config_path.resolve())

    with _AUTOGEN_LOCK:
        if _AUTOGEN_CANCEL is not None:
            return {"error": "Auto-generation is already running"}
        _AUTOGEN_CANCEL = threading.Event()
        _AUTOGEN_STATUS = {
            "running": True,
            "identity_id": int(identity_id),
            "matched": 0,
            "attempts": 0,
            "target_count": int(target_count),
            "max_attempts": int(max_attempts),
            "message": "Starting...",
        }

    try:
        label = _get_identity_label(config, identity_id)
        if not label:
            _set_status(message="Identity has no label set.", running=False)
            return {"error": "Identity has no label. Set a label first."}

        centroid = _get_identity_centroid(config, identity_id)
        if centroid is None:
            _set_status(message="Identity has no centroid.", running=False)
            return {"error": "Identity has no centroid. Run a scan first."}

        ag = config.get("auto_generate", {})
        webbduck_url = str(ag.get("webbduck_url", "http://localhost:8020"))
        base_model = str(ag.get("base_model", config.get("kohya_base_model", "")))
        if not base_model:
            _set_status(message="No base_model configured for auto-generation.", running=False)
            return {"error": "auto_generate.base_model or kohya_base_model must be set in config."}

        steps = int(ag.get("steps", 30))
        cfg = float(ag.get("cfg", 7.5))
        width = int(ag.get("width", 1024))
        height = int(ag.get("height", 1024))
        scheduler = str(ag.get("scheduler", "UniPC"))
        second_pass_model = str(ag.get("second_pass_model", "None"))
        negative_prompt = str(ag.get("negative_prompt",
            "low quality, blurry, bad anatomy, disfigured, extra limbs, bad hands"))
        mode = str(config.get("mode", "realism"))
        eps_r = float(ag.get("assign_eps_realism", config.get("assign_eps_realism", 0.27)))
        eps_a = float(ag.get("assign_eps_anime", config.get("assign_eps_anime", 0.39)))
        if mode == "anime":
            assign_eps = eps_a
        elif mode == "hybrid":
            assign_eps = (eps_r + eps_a) / 2.0
        else:
            assign_eps = eps_r

        output_dir = Path(config["output_folder"]).resolve() / "autogen"
        output_dir.mkdir(parents=True, exist_ok=True)

        embedder = FaceEmbedder(config)
        templates = _load_prompt_templates(config)

        matched = 0
        attempts = 0

        while matched < target_count and attempts < max_attempts:
            if _AUTOGEN_CANCEL and _AUTOGEN_CANCEL.is_set():
                _set_status(message="Cancelled by user.")
                break

            attempts += 1
            prompt = _build_prompt(label, templates)
            _set_status(
                matched=matched, attempts=attempts,
                message=f"Attempt {attempts}/{max_attempts}, generating...",
            )
            if progress_callback:
                progress_callback(matched, attempts, prompt)

            gen_path = _generate_image_via_webbduck(
                prompt=prompt,
                negative_prompt=negative_prompt,
                webbduck_url=webbduck_url,
                base_model=base_model,
                cfg=cfg,
                steps=steps,
                width=width,
                height=height,
                scheduler=scheduler,
                second_pass_model=second_pass_model,
            )
            if gen_path is None:
                log.warning("Generation returned no image (attempt %d)", attempts)
                continue

            gen_file = Path(gen_path)
            if not gen_file.exists():
                log.warning("Generated file not found: %s", gen_file)
                continue

            _set_status(message=f"Attempt {attempts}: analyzing...")

            if _match_to_identity(gen_file, embedder, centroid, assign_eps):
                matched += 1
                _add_to_dataset(config, gen_file, identity_id)
                log.info("MATCH #%d: %s (attempt %d)", matched, gen_file.name, attempts)
            else:
                try:
                    gen_file.unlink()
                except Exception:
                    pass
                log.info("No match (attempt %d): deleted %s", attempts, gen_file.name)

        _set_status(
            matched=matched, attempts=attempts, running=False,
            message=f"Done. Matched {matched}/{target_count} in {attempts} attempts.",
        )
        return {
            "matched": matched,
            "attempts": attempts,
            "target_count": target_count,
            "identity_id": int(identity_id),
            "label": label,
        }

    except Exception as exc:
        _set_status(message=f"Error: {exc}", running=False)
        log.exception("Auto-generation failed")
        return {"error": str(exc)}
    finally:
        with _AUTOGEN_LOCK:
            _AUTOGEN_CANCEL = None


def cancel_auto_generate() -> dict:
    with _AUTOGEN_LOCK:
        if _AUTOGEN_CANCEL is None:
            return {"cancelled": False, "message": "No auto-generation running."}
        _AUTOGEN_CANCEL.set()
        _AUTOGEN_STATUS["message"] = "Cancelling..."
        return {"cancelled": True}


def get_auto_generate_status() -> dict:
    with _AUTOGEN_LOCK:
        if not _AUTOGEN_STATUS.get("running"):
            return {"running": False, "message": _AUTOGEN_STATUS.get("message", "Idle")}
        return dict(_AUTOGEN_STATUS)


def _set_status(matched=None, attempts=None, message=None, running=None):
    with _AUTOGEN_LOCK:
        if matched is not None:
            _AUTOGEN_STATUS["matched"] = matched
        if attempts is not None:
            _AUTOGEN_STATUS["attempts"] = attempts
        if message is not None:
            _AUTOGEN_STATUS["message"] = message
        if running is not None:
            _AUTOGEN_STATUS["running"] = running
