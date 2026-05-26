from __future__ import annotations

import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from .utils import ensure_clean_dir, to_json_file, to_jsonl_file


def export_metadata(
    conn: sqlite3.Connection,
    *,
    output_folder: Path,
    input_folder: Path,
    path_scope: Path | None = None,
    overwrite: bool = True,
) -> dict:
    ensure_clean_dir(output_folder, overwrite=overwrite)

    image_rows = conn.execute(
        """
        SELECT path, sha256, status, identity_id, size_bytes, mtime, created_at, updated_at, last_seen_at
        FROM images
        ORDER BY path ASC
        """
    ).fetchall()
    identity_rows = conn.execute(
        """
        SELECT id, label, member_count, created_at, updated_at
        FROM identities
        ORDER BY id ASC
        """
    ).fetchall()

    manifest = []
    scope = path_scope.resolve() if path_scope is not None else None
    for row in image_rows:
        absolute = Path(row["path"]).resolve()
        if not absolute.exists():
            continue
        if scope is not None:
            try:
                absolute.relative_to(scope)
            except ValueError:
                continue
        try:
            relative = absolute.relative_to(input_folder)
            relative_str = str(relative)
        except ValueError:
            relative_str = str(absolute)
        manifest.append(
            {
                "path": str(absolute),
                "relative_path": relative_str,
                "sha256": row["sha256"],
                "status": row["status"],
                "identity_id": row["identity_id"],
                "size_bytes": row["size_bytes"],
                "mtime": row["mtime"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_seen_at": row["last_seen_at"],
            }
        )

    identities = [
        {
            "identity_id": row["id"],
            "label": row["label"],
            "member_count": row["member_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in identity_rows
    ]

    to_json_file(output_folder / "manifest.json", manifest)
    to_json_file(output_folder / "identities.json", identities)
    return {"images": len(manifest), "identities": len(identities)}


def export_identity_views(
    conn: sqlite3.Connection,
    *,
    output_folder: Path,
    link_mode: str = "none",
    path_scope: Path | None = None,
    overwrite: bool = True,
) -> dict:
    link_mode = link_mode.lower().strip()
    if link_mode not in {"none", "symlink", "hardlink", "copy"}:
        raise ValueError("link_mode must be one of: none, symlink, hardlink, copy")

    views_root = output_folder / "identities"
    ensure_clean_dir(views_root, overwrite=overwrite)

    if link_mode == "none":
        return {"linked": 0}

    rows = conn.execute(
        """
        SELECT path, identity_id, status
        FROM images
        ORDER BY identity_id ASC, path ASC
        """
    ).fetchall()
    scope = path_scope.resolve() if path_scope is not None else None
    linked = 0
    for row in rows:
        status = str(row["status"])
        source = Path(row["path"]).resolve()
        if not source.exists():
            continue
        if scope is not None:
            try:
                source.relative_to(scope)
            except ValueError:
                continue
        if status != "assigned" or row["identity_id"] is None:
            target_dir = views_root / "unassigned"
        else:
            target_dir = views_root / f"identity_{int(row['identity_id'])}"
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = _next_available_target(target_dir / source.name)
        _materialize_link(source, destination, link_mode)
        linked += 1
    return {"linked": linked}


def export_lora_dataset(
    conn: sqlite3.Connection,
    *,
    output_folder: Path,
    link_mode: str = "hardlink",
    min_images: int = 5,
    identity_ids: list[int] | None = None,
    overwrite: bool = True,
    face_crop: bool = False,
    face_crop_size: int = 1024,
    face_crop_padding: float = 0.35,
    face_crop_model_name: str = "buffalo_l",
    face_crop_use_gpu: bool = False,
) -> dict:
    root = output_folder / "lora_export"
    ensure_clean_dir(root, overwrite=overwrite)
    crop_size = max(256, int(face_crop_size))
    crop_padding = max(0.0, float(face_crop_padding))
    cropper = _FaceCropper(
        enabled=bool(face_crop),
        target_size=crop_size,
        padding=crop_padding,
        model_name=str(face_crop_model_name or "buffalo_l"),
        use_gpu=bool(face_crop_use_gpu),
    )

    selected_ids = sorted({int(v) for v in (identity_ids or []) if int(v) > 0})
    if selected_ids:
        placeholders = ",".join(["?"] * len(selected_ids))
        identities = conn.execute(
            f"""
            SELECT id, label, member_count
            FROM identities
            WHERE id IN ({placeholders})
            ORDER BY member_count DESC, id ASC
            """,
            tuple(selected_ids),
        ).fetchall()
    else:
        identities = conn.execute(
            """
            SELECT id, label, member_count
            FROM identities
            WHERE member_count >= ?
            ORDER BY member_count DESC, id ASC
            """,
            (int(min_images),),
        ).fetchall()

    identity_count = 0
    image_count = 0
    exported_identity_ids: list[int] = []
    for identity in identities:
        identity_id = int(identity["id"])
        label = identity["label"] or f"identity_{identity_id}"
        identity_dir = root / f"identity_{identity_id}"
        images_dir = identity_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_rows = conn.execute(
            """
            SELECT path
            FROM images
            WHERE identity_id = ? AND status = 'assigned'
            ORDER BY path ASC
            """,
            (identity_id,),
        ).fetchall()
        metadata_rows = []
        for row in image_rows:
            source = Path(row["path"])
            if not source.exists():
                continue
            destination = _next_available_target(
                images_dir / (f"{source.stem}.jpg" if cropper.enabled else source.name)
            )
            transform = "none"
            if cropper.enabled:
                cropper.write_processed(source=source, destination=destination)
                transform = f"face_crop_{cropper.target_size}"
            else:
                _materialize_link(source, destination, link_mode)
            caption_path = destination.with_suffix(".txt")
            caption_path.write_text(f"{label}\n", encoding="utf-8")
            metadata_rows.append(
                {
                    "file_name": destination.name,
                    "source_path": str(source),
                    "caption": label,
                    "identity_id": identity_id,
                    "transform": transform,
                }
            )
            image_count += 1

        to_jsonl_file(identity_dir / "metadata.jsonl", metadata_rows)
        to_json_file(
            identity_dir / "identity.json",
            {
                "identity_id": identity_id,
                "label": label,
                "member_count": int(identity["member_count"]),
                "link_mode": link_mode,
            },
        )
        identity_count += 1
        exported_identity_ids.append(identity_id)

    return {
        "identities": identity_count,
        "images": image_count,
        "caption_mode": "identity_token",
        "image_preprocess": (
            f"face_crop_{cropper.target_size}" if cropper.enabled else "none"
        ),
        "requested_identity_ids": selected_ids,
        "exported_identity_ids": exported_identity_ids,
    }


def _next_available_target(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    idx = 1
    while True:
        candidate = destination.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _materialize_link(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return

    if mode == "symlink":
        try:
            os.symlink(source, destination)
            return
        except OSError:
            try:
                os.link(source, destination)
                return
            except OSError:
                shutil.copy2(source, destination)
                return

    raise ValueError(f"Unsupported link mode: {mode}")


@dataclass
class _FaceCropper:
    enabled: bool
    target_size: int
    padding: float
    model_name: str
    use_gpu: bool
    _analyzer: Any | None = None

    def write_processed(self, *, source: Path, destination: Path) -> None:
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Could not read image for face crop: {source}")
        cropped = self._crop_face(image)
        ok = cv2.imwrite(str(destination), cropped)
        if not ok:
            raise RuntimeError(f"Failed to write face-cropped image: {destination}")

    def _crop_face(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return self._center_square(image_bgr)
        bbox = self._detect_primary_face_bbox(image_bgr)
        if bbox is None:
            return self._center_square(image_bgr)
        x1, y1, x2, y2 = bbox
        fw = max(1.0, x2 - x1)
        fh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        side = max(fw, fh) * (1.0 + (2.0 * self.padding))
        side = max(32.0, min(side, float(min(w, h))))
        left = int(round(cx - (side * 0.5)))
        top = int(round(cy - (side * 0.5)))
        left = max(0, min(left, int(w - side)))
        top = max(0, min(top, int(h - side)))
        side_i = int(max(1, min(side, w - left, h - top)))
        crop = image_bgr[top : top + side_i, left : left + side_i]
        if crop.size == 0:
            crop = self._center_square(image_bgr)
        return cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def _center_square(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        side = min(h, w)
        top = max(0, (h - side) // 2)
        left = max(0, (w - side) // 2)
        crop = image_bgr[top : top + side, left : left + side]
        if crop.size == 0:
            crop = image_bgr
        return cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def _detect_primary_face_bbox(self, image_bgr: np.ndarray) -> tuple[float, float, float, float] | None:
        analyzer = self._get_analyzer()
        if analyzer is None:
            return None
        try:
            faces = analyzer.get(image_bgr)
        except Exception:
            return None
        if not faces:
            return None
        face = max(faces, key=lambda item: float(max(0.0, item.bbox[2] - item.bbox[0])) * float(max(0.0, item.bbox[3] - item.bbox[1])))
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        return x1, y1, x2, y2

    def _get_analyzer(self):
        if self._analyzer is not None:
            return self._analyzer
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "Face-crop export requires InsightFace. Install dependencies from requirements.txt."
            ) from exc
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        analyzer = FaceAnalysis(name=self.model_name, providers=providers)
        analyzer.prepare(ctx_id=0 if self.use_gpu else -1, det_size=(640, 640))
        self._analyzer = analyzer
        return self._analyzer
