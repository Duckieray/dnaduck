from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

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
) -> dict:
    root = output_folder / "lora_export"
    ensure_clean_dir(root, overwrite=overwrite)

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
            destination = _next_available_target(images_dir / source.name)
            _materialize_link(source, destination, link_mode)
            caption_path = destination.with_suffix(".txt")
            caption_path.write_text(f"{label}\n", encoding="utf-8")
            metadata_rows.append(
                {
                    "file_name": destination.name,
                    "source_path": str(source),
                    "caption": label,
                    "identity_id": identity_id,
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
