from __future__ import annotations

import json
import logging
import random
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .embedder import FaceEmbedder, LoadedImage

log = logging.getLogger("dnaduck.autogen")


# ── Prompt option accuracy stats ──────────────────────────────────────

def _stats_path(config_path: Path) -> Path:
    return config_path.resolve().parent / ".dnaduck_autogen_stats.json"


def _load_autogen_stats(config_path: Path) -> dict:
    sp = _stats_path(config_path)
    if not sp.exists():
        return {"prompt_stats": {}, "version": 1, "last_updated": ""}
    try:
        with open(sp, "r") as f:
            return json.load(f)
    except Exception:
        return {"prompt_stats": {}, "version": 1, "last_updated": ""}


def _save_autogen_stats(config_path: Path, stats: dict):
    stats["last_updated"] = datetime.now(timezone.utc).isoformat()
    sp = _stats_path(config_path)
    try:
        with open(sp, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as exc:
        log.warning("Failed to save autogen stats: %s", exc)


def _merge_autogen_stats(
    stats: dict,
    template_cats: list[str],
    attempt_counts: dict[str, dict[str, int]],
    match_counts: dict[str, dict[str, int]],
    assign_eps_used: float,
):
    ps = stats.setdefault("prompt_stats", {})
    for cat in template_cats:
        cat_stats = ps.setdefault(cat, {})
        for text in attempt_counts.get(cat, {}):
            a = attempt_counts[cat][text]
            m = match_counts[cat].get(text, 0)
            entry = cat_stats.setdefault(text, {"attempts": 0, "matches": 0, "best_eps": None})
            entry["attempts"] += a
            entry["matches"] += m
            if m > 0:
                if entry["best_eps"] is None or assign_eps_used < entry["best_eps"]:
                    entry["best_eps"] = assign_eps_used
                entry["last_match_eps"] = assign_eps_used


def _autogen_stats_readable(config_path: Path) -> list[dict]:
    """Return per-option accuracy stats sorted by accuracy ascending."""
    stats = _load_autogen_stats(config_path)
    ps = stats.get("prompt_stats", {})
    rows = []
    for cat, options in ps.items():
        for text, data in options.items():
            a = data.get("attempts", 0)
            m = data.get("matches", 0)
            pct = (m / a * 100) if a > 0 else 0.0
            rows.append({
                "category": cat,
                "text": text,
                "attempts": a,
                "matches": m,
                "accuracy": round(pct, 1),
                "best_eps": data.get("best_eps"),
                "last_match_eps": data.get("last_match_eps"),
            })
    rows.sort(key=lambda r: r["accuracy"])
    return rows

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


DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "shot": ["ultra closeup", "portrait", "3/4 shot", "full body", "side profile"],
    "pose": ["facing camera", "facing to the side", "standing", "sitting"],
    "hair": ["hair up", "hair down"],
    "setting": ["in a coffee shop", "in a forest", "hiking on a trail", "in a park",
                "on a city street", "in a studio", "in a garden", "at the beach"],
    "clothing": ["casual clothes", "formal wear", "summer dress", "leather jacket", "uniform"],
    "style": ["", "cinematic lighting", "soft natural lighting", "professional photography"],
}


def _load_prompt_templates(config: dict) -> dict[str, list[tuple[str, float]]]:
    raw = config.get("auto_generate", {})
    templates = raw.get("prompt_templates", {})
    if not templates:
        return {k: _resolve_weighted_list(v) for k, v in DEFAULT_TEMPLATES.items()}
    return {k: _resolve_weighted_list(v) for k, v in templates.items() if _resolve_weighted_list(v)}


PROMPT_ORDER = ["shot", "orientation", "posture", "expression", "hair", "setting", "clothing", "style"]


def _build_prompt(character_label: str, templates: dict[str, list[tuple[str, float]]], basic_prompt: str = "") -> tuple[str, dict[str, str]]:
    choices = {}
    for cat in PROMPT_ORDER:
        if cat in templates:
            choices[cat] = _weighted_choice(templates[cat])
    resolved_label = character_label
    if basic_prompt:
        basic = basic_prompt.replace("{trigger}", character_label)
        parts = [resolved_label]
        for cat in PROMPT_ORDER:
            if cat in choices and choices[cat]:
                parts.append(choices[cat])
        parts.append("detailed face, high quality")
        return f"{basic}, {', '.join(parts)}", choices
    parts = [resolved_label]
    for cat in PROMPT_ORDER:
        if cat in choices and choices[cat]:
            parts.append(choices[cat])
    parts.append("detailed face, high quality")
    return ", ".join(parts), choices


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
    loras: list | None = None,
    embeddings: list | None = None,
    identity_adapter: dict | None = None,
) -> str | None:
    import json
    url = f"{webbduck_url.rstrip('/')}/test"
    params = {
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
    }
    if loras:
        params["loras"] = json.dumps(loras)
        log.info("Sending LoRAs to WebbDuck: %s", params["loras"])
    if embeddings:
        params["embeddings"] = json.dumps(embeddings)
        log.info("Sending embeddings to WebbDuck: %s", params["embeddings"])
    if identity_adapter and identity_adapter.get("enabled"):
        # Resolve reference image paths to absolute paths before sending
        adapter_cfg = dict(identity_adapter)
        dnaduck_root = Path(__file__).resolve().parent.parent
        refs = adapter_cfg.get("reference_images", [])
        if refs:
            abs_refs = []
            for ref in refs:
                p = Path(ref)
                if not p.is_absolute():
                    p = dnaduck_root / p
                abs_refs.append(str(p.resolve()))
            adapter_cfg["reference_images"] = abs_refs
        params["identity_adapter"] = json.dumps(adapter_cfg)
        log.info("Sending identity_adapter to WebbDuck: %s", params["identity_adapter"])
        
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
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
    matching_config: dict | None = None,
    anchor_embeddings: list[np.ndarray] | None = None,
) -> tuple[str, np.ndarray | None, float, float]:
    array_bgr = cv2.imread(str(image_path))
    if array_bgr is None:
        log.warning("Autogen: cv2.imread returned None for %s", image_path)
        return "reject", None, 0.0, 1.0
    loaded = LoadedImage(path=image_path, array_bgr=array_bgr)
    result = embedder.extract([loaded])
    if str(image_path) not in [str(p) for p in result.embedded_paths]:
        log.warning("Autogen: no face detected in %s", image_path)
        return "reject", None, 0.0, 1.0

    paths_str = [str(p) for p in result.embedded_paths]
    idx = paths_str.index(str(image_path))
    vector = result.embeddings[idx]

    if matching_config and matching_config.get("mode") == "anchor_set" and anchor_embeddings:
        distances = []
        for anchor in anchor_embeddings:
            sim = float(np.clip(np.dot(vector, anchor), -1.0, 1.0))
            distances.append(1.0 - sim)
        
        median_dist = float(np.median(distances))
        max_dist = float(np.max(distances))
        
        accept_threshold = float(matching_config.get("accept_threshold", 0.22))
        hard_max_threshold = float(matching_config.get("hard_max_threshold", 0.26))
        
        review_band = matching_config.get("review_band", {})
        review_min = float(review_band.get("min", 0.22))
        review_max = float(review_band.get("max", 0.26))
        
        log.warning(
            "Autogen anchor match check: file=%s median_dist=%.4f max_dist=%.4f accept_thresh=%.4f max_thresh=%.4f",
            image_path.name, median_dist, max_dist, accept_threshold, hard_max_threshold
        )
        
        if median_dist <= accept_threshold and max_dist <= hard_max_threshold:
            return "accept", vector, 1.0 - median_dist, median_dist
        elif review_min <= median_dist <= review_max:
            return "review", vector, 1.0 - median_dist, median_dist
        else:
            return "reject", vector, 1.0 - median_dist, median_dist
    else:
        cos_sim = float(np.clip(np.dot(vector, identity_centroid), -1.0, 1.0))
        cos_dist = 1.0 - cos_sim

        log.warning(
            "Autogen match check: file=%s cos_sim=%.4f cos_dist=%.4f threshold=%.4f matched=%s",
            image_path.name,
            cos_sim,
            cos_dist,
            assign_eps,
            cos_dist <= assign_eps,
        )

        matched = cos_dist <= assign_eps
        return "accept" if matched else "reject", vector, cos_sim, cos_dist


def _add_to_dataset(config: dict, image_path: Path, identity_id: int | None, embedding: np.ndarray, status: str = "assigned") -> None:
    import hashlib
    from .database import (
        open_database,
        serialize_embedding,
        upsert_image,
        rebuild_identity_stats,
    )

    db_path = Path(config["database_path"])
    conn = open_database(db_path)
    try:
        stat = image_path.stat()
        sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        upsert_image(
            conn,
            path=image_path,
            sha256=sha,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            status=status,
            identity_id=identity_id,
            embedding=embedding,
            face_score=None,
        )
        if identity_id is not None and status == "assigned":
            rebuild_identity_stats(conn, identity_id)
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


def init_auto_generate_status(
    identity_id: int,
    target_count: int = 50,
    max_attempts: int = 500,
) -> None:
    """Set initial status before the thread starts, so get_auto_generate_status()
    never sees stale data from a previous run."""
    global _AUTOGEN_CANCEL, _AUTOGEN_STATUS
    with _AUTOGEN_LOCK:
        if _AUTOGEN_CANCEL is not None:
            return  # already running, don't clobber
        _AUTOGEN_CANCEL = threading.Event()
        _AUTOGEN_STATUS = {
            "running": True,
            "identity_id": int(identity_id),
            "matched": 0,
            "attempts": 0,
            "target_count": int(target_count),
            "max_attempts": int(max_attempts),
            "message": "Starting...",
            "last_prompt": "",
            "runtime_config": None,
        }


def run_auto_generate(
    config_path: Path,
    identity_id: int,
    target_count: int = 50,
    max_attempts: int = 500,
    progress_callback=None,
    assign_eps_realism: float | None = None,
    assign_eps_anime: float | None = None,
    target_identity_id: int | None = None,
) -> dict:
    global _AUTOGEN_CANCEL, _AUTOGEN_STATUS

    from .utils import load_config, resolve_runtime_paths

    config = load_config(config_path.resolve())
    config = resolve_runtime_paths(config, config_path.resolve())

    # Status was already initialized by init_auto_generate_status() in the
    # service layer — just verify the cancel flag is set.
    if _AUTOGEN_CANCEL is None:
        return {"error": "Auto-generation not initialized properly"}

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
        webbduck_url = str(ag.get("webbduck_url", "http://webbduck.theducklabs.com"))
        webbduck_output_dir = str(ag.get("webbduck_output_dir", "")).strip()
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
        basic_prompt = str(ag.get("basic_prompt", ""))
        # New format: loras = [{name, weight}, ...]
        # Legacy compat: fall back to lora_name / lora_weight if loras list absent
        loras_cfg: list[dict] = []
        if ag.get("loras") and isinstance(ag["loras"], list):
            loras_cfg = [e for e in ag["loras"] if isinstance(e, dict) and e.get("name")]
        elif ag.get("lora_name"):
            try:
                loras_cfg = [{"name": str(ag["lora_name"]), "weight": float(ag.get("lora_weight", 1.0))}]
            except (ValueError, TypeError):
                loras_cfg = [{"name": str(ag["lora_name"]), "weight": 1.0}]
        embeddings_cfg: list[dict] = []
        if ag.get("embeddings") and isinstance(ag["embeddings"], list):
            embeddings_cfg = [e for e in ag["embeddings"] if isinstance(e, dict) and e.get("name")]
            
        identity_adapter = ag.get("identity_adapter", {})
        
        mode = str(config.get("mode", "realism"))
        eps_r = float(ag.get("assign_eps_realism", config.get("assign_eps_realism", 0.60)))
        eps_a = float(ag.get("assign_eps_anime", config.get("assign_eps_anime", 0.75)))
        if assign_eps_realism is not None:
            eps_r = float(assign_eps_realism)
        if assign_eps_anime is not None:
            eps_a = float(assign_eps_anime)
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

        matching_config = config.get("matching", {})
        anchor_embeddings = []
        if matching_config.get("mode") == "anchor_set":
            dnaduck_root = Path(__file__).resolve().parent.parent
            for anchor_path_str in matching_config.get("anchor_images", []):
                anchor_path = Path(anchor_path_str)
                if not anchor_path.is_absolute():
                    anchor_path = (dnaduck_root / anchor_path).resolve()
                else:
                    anchor_path = anchor_path.resolve()
                if not anchor_path.exists():
                    log.warning(f"Anchor image not found: {anchor_path}")
                    continue
                a_bgr = cv2.imread(str(anchor_path))
                if a_bgr is not None:
                    ld = LoadedImage(path=anchor_path, array_bgr=a_bgr)
                    res = embedder.extract([ld])
                    if res.embeddings:
                        anchor_embeddings.append(res.embeddings[0])
            if not anchor_embeddings:
                log.warning("Anchor set mode enabled but no valid anchor embeddings could be extracted. Falling back to centroid.")

        log.info(
            "Auto-generate config: basic_prompt=%r loras=%r embeddings=%r",
            basic_prompt, loras_cfg, embeddings_cfg,
        )

        matched = 0
        attempts = 0
        template_cats = list(templates.keys())
        match_counts: dict[str, dict[str, int]] = {cat: {} for cat in template_cats}
        attempt_counts: dict[str, dict[str, int]] = {cat: {} for cat in template_cats}

        def _sample_with_distribution(cur_templates: dict) -> tuple[str, dict[str, str]]:
            adjusted = {cat: list(cur_templates[cat]) for cat in template_cats}
            for cat in template_cats:
                items = adjusted[cat]
                if not items:
                    continue
                counts = match_counts[cat]
                max_count = max(counts.values()) if counts else 0
                if max_count < 2:
                    continue
                boosted = []
                for text, base_w in items:
                    c = counts.get(text, 0)
                    gap = max_count - c
                    boost = 1.0 + float(gap) * 2.0
                    boosted.append((text, base_w * boost))
                adjusted[cat] = boosted
            p, c = _build_prompt(label, adjusted, basic_prompt=basic_prompt)
            return p, c

        def _pick_underperformers() -> list[tuple[str, str, int]]:
            result = []
            for cat in template_cats:
                attempts_cat = attempt_counts[cat]
                matches_cat = match_counts[cat]
                for text in attempts_cat:
                    a = attempts_cat.get(text, 0)
                    m = matches_cat.get(text, 0)
                    if a >= 5 and m == 0:
                        result.append((cat, text, a))
            result.sort(key=lambda x: -x[2])
            return result

        def _check_targets_done():
            return matched >= target_count or attempts >= max_attempts

        while matched < target_count and attempts < max_attempts:
            if _AUTOGEN_CANCEL and _AUTOGEN_CANCEL.is_set():
                _set_status(message="Cancelled by user.")
                break

            attempts += 1
            prompt, choices = _sample_with_distribution(templates)
            _set_status(
                matched=matched, attempts=attempts,
                message=f"Attempt {attempts}/{max_attempts}, generating...",
                last_prompt=prompt,
                runtime_config={
                    "basic_prompt": basic_prompt,
                    "loras": loras_cfg,
                    "embeddings": embeddings_cfg,
                },
            )
            if progress_callback:
                progress_callback(matched, attempts, prompt)

            log.info("Prompt [%d/%d]: %s", attempts, max_attempts, prompt)
            if loras_cfg:
                log.info("  LoRAs: %s", loras_cfg)
            if embeddings_cfg:
                log.info("  Embeddings: %s", embeddings_cfg)

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
                loras=loras_cfg,
                embeddings=embeddings_cfg,
                identity_adapter=identity_adapter,
            )
            if gen_path is None:
                log.warning("Generation returned no image (attempt %d)", attempts)
                continue

            gen_file = Path(gen_path)
            if not gen_file.exists() and webbduck_output_dir:
                # WebbDuck returns web paths like "outputs/2024-07-01/0.png".
                # Resolve them against webbduck_output_dir to get the real path.
                try:
                    if "outputs/" in str(gen_path):
                        rel = str(gen_path).split("outputs/", 1)[1].lstrip("/")
                        resolved = Path(webbduck_output_dir) / rel
                        if resolved.exists():
                            gen_file = resolved
                except Exception:
                    pass
            if not gen_file.exists():
                log.warning("Generated file not found: %s", gen_file)
                continue

            _set_status(message=f"Attempt {attempts}: analyzing...")

            for cat, val in choices.items():
                ac = attempt_counts[cat]
                ac[val] = ac.get(val, 0) + 1

            match_status, vector, cos_sim, cos_dist = _match_to_identity(
                gen_file, embedder, centroid, assign_eps, matching_config, anchor_embeddings
            )
            if match_status == "accept":
                matched += 1
                save_identity_id = target_identity_id if target_identity_id is not None else identity_id
                log.info(
                    "Adding match to identity_id=%s (target was %s, fallback=%s)",
                    save_identity_id, target_identity_id, identity_id,
                )
                _add_to_dataset(config, gen_file, save_identity_id, vector, status="assigned")
                for cat, val in choices.items():
                    mc = match_counts[cat]
                    mc[val] = mc.get(val, 0) + 1
                log.info(
                    "MATCH #%d: %s (attempt %d)",
                    matched, gen_file.name, attempts,
                )
            elif match_status == "review":
                save_identity_id = target_identity_id if target_identity_id is not None else identity_id
                log.info("Match hit review band: %s (attempt %d)", gen_file.name, attempts)
                _add_to_dataset(config, gen_file, save_identity_id, vector, status="unassigned")
            else:
                try:
                    gen_file.unlink()
                except Exception:
                    pass
                log.info("No match (attempt %d): deleted %s", attempts, gen_file.name)

        log.info(
            "Autogen distribution: %s",
            {cat: match_counts.get(cat, {}) for cat in template_cats},
        )

        # ── Catch-up pass for hard-to-match options ────────────────────
        if matched < target_count:
            underperformers = _pick_underperformers()
            if underperformers:
                log.warning(
                    "Underperforming options detected (5+ attempts, 0 matches): %s",
                    underperformers,
                )
                # Load historical stats to use smarter starting eps
                historical_stats = _load_autogen_stats(config_path)
                historical_ps = historical_stats.get("prompt_stats", {})
                for u_cat, u_text, u_attempts in underperformers:
                    if matched >= target_count:
                        break
                    # Check if this option has ever matched before, and use its best_eps
                    entry = historical_ps.get(u_cat, {}).get(u_text, {})
                    historical_best_eps = entry.get("best_eps")
                    if historical_best_eps is not None:
                        start_eps = min(historical_best_eps + 0.05, 0.80)
                        log.info(
                            "Catch-up for %s/%s: %d failed attempts, historical best_eps=%.3f, starting at %.3f",
                            u_cat, u_text, u_attempts, historical_best_eps, start_eps,
                        )
                    else:
                        start_eps = assign_eps
                        log.info(
                            "Catch-up for %s/%s: %d failed attempts, no history, starting at %.3f",
                            u_cat, u_text, u_attempts, start_eps,
                        )
                    current_eps = start_eps
                    while matched < target_count and not _check_targets_done():
                        if _AUTOGEN_CANCEL and _AUTOGEN_CANCEL.is_set():
                            break
                        if current_eps > 0.80:
                            log.warning("Catch-up eps=%.3f > 0.80, giving up on %s/%s", current_eps, u_cat, u_text)
                            break
                        batch_matched = 0
                        for batch_attempt in range(5):
                            if matched >= target_count or _check_targets_done():
                                break
                            if _AUTOGEN_CANCEL and _AUTOGEN_CANCEL.is_set():
                                break
                            attempts += 1
                            prompt, choices = _sample_with_distribution(templates)
                            _set_status(
                                matched=matched, attempts=attempts,
                                message=f"Catch-up {u_cat}/{u_text} eps={current_eps:.2f}: {prompt[:50]}...",
                                last_prompt=prompt,
                            )
                            log.info("Catch-up [eps=%.3f] %s/%s: %s", current_eps, u_cat, u_text, prompt)
                            gen_path = _generate_image_via_webbduck(
                                prompt=prompt, negative_prompt=negative_prompt,
                                webbduck_url=webbduck_url, base_model=base_model,
                                cfg=cfg, steps=steps, width=width, height=height,
                                scheduler=scheduler, second_pass_model=second_pass_model,
                                loras=loras_cfg, embeddings=embeddings_cfg,
                                identity_adapter=identity_adapter,
                            )
                            if gen_path is None:
                                continue
                            gen_file = Path(gen_path)
                            if not gen_file.exists() and webbduck_output_dir:
                                try:
                                    if "outputs/" in str(gen_path):
                                        rel = str(gen_path).split("outputs/", 1)[1].lstrip("/")
                                        resolved = Path(webbduck_output_dir) / rel
                                        if resolved.exists():
                                            gen_file = resolved
                                except Exception:
                                    pass
                            if not gen_file.exists():
                                continue
                            for cat, val in choices.items():
                                ac = attempt_counts[cat]
                                ac[val] = ac.get(val, 0) + 1
                            match_status, vector, cos_sim, cos_dist = _match_to_identity(
                                gen_file, embedder, centroid, current_eps, matching_config, anchor_embeddings
                            )
                            if match_status == "accept":
                                matched += 1
                                batch_matched += 1
                                save_identity_id = target_identity_id if target_identity_id is not None else identity_id
                                _add_to_dataset(config, gen_file, save_identity_id, vector, status="assigned")
                                for cat, val in choices.items():
                                    mc = match_counts[cat]
                                    mc[val] = mc.get(val, 0) + 1
                                log.info("Catch-up MATCH #%d: %s (eps=%.3f)", matched, gen_file.name, current_eps)
                            elif match_status == "review":
                                save_identity_id = target_identity_id if target_identity_id is not None else identity_id
                                log.info("Catch-up match hit review band: %s (eps=%.3f)", gen_file.name, current_eps)
                                _add_to_dataset(config, gen_file, save_identity_id, vector, status="unassigned")
                            else:
                                try:
                                    gen_file.unlink()
                                except Exception:
                                    pass
                        if batch_matched > 0:
                            log.info("Catch-up batch for %s/%s: %d matches at eps=%.3f", u_cat, u_text, batch_matched, current_eps)
                        else:
                            current_eps = min(current_eps + 0.05, 0.80)
                            log.info("Catch-up batch for %s/%s: 0 matches, raising eps to %.3f", u_cat, u_text, current_eps)

        # ── Persist accuracy stats ────────────────────────────────────
        stats = _load_autogen_stats(config_path)
        _merge_autogen_stats(stats, template_cats, attempt_counts, match_counts, assign_eps)
        _save_autogen_stats(config_path, stats)

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
        if not _AUTOGEN_STATUS:
            return {
                "running": False,
                "message": "Idle",
                "last_prompt": "",
                "runtime_config": None,
            }
        return dict(_AUTOGEN_STATUS)


def _set_status(matched=None, attempts=None, message=None, running=None, last_prompt=None, runtime_config=None):
    with _AUTOGEN_LOCK:
        if matched is not None:
            _AUTOGEN_STATUS["matched"] = matched
        if attempts is not None:
            _AUTOGEN_STATUS["attempts"] = attempts
        if message is not None:
            _AUTOGEN_STATUS["message"] = message
        if running is not None:
            _AUTOGEN_STATUS["running"] = running
        if last_prompt is not None:
            _AUTOGEN_STATUS["last_prompt"] = last_prompt
        if runtime_config is not None:
            _AUTOGEN_STATUS["runtime_config"] = runtime_config
