from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .cluster import cluster_faces
from .database import (
    IdentityRecord,
    create_identity,
    create_scan,
    fetch_existing_image_index,
    fetch_identity_records,
    finalize_scan,
    open_database,
    rebuild_identity_stats,
    touch_existing_image,
    update_image_status,
    upsert_image,
)
from .embedder import FaceEmbedder
from .exporter import export_identity_views, export_lora_dataset, export_metadata
from .utils import (
    discover_image_paths,
    get_logger,
    load_images_from_paths,
    sha256_file,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ScanResult:
    scan_id: str
    discovered_count: int
    processed_count: int
    assigned_count: int
    noise_count: int
    no_face_count: int
    identity_count: int
    metadata_images: int
    metadata_identities: int
    linked_images: int


def resolve_assign_eps(config: dict) -> float:
    mode = str(config.get("mode", "realism")).lower().strip()
    realism = float(config.get("assign_eps_realism", config.get("eps_realism", 0.60)))
    anime = float(config.get("assign_eps_anime", config.get("eps_anime", 0.75)))
    hybrid = float(config.get("assign_eps_hybrid", (realism + anime) / 2.0))
    if mode == "realism":
        return realism
    if mode == "anime":
        return anime
    if mode == "hybrid":
        return hybrid
    raise ValueError(f"Unsupported mode '{mode}'. Expected realism | anime | hybrid.")


def run_scan(config: dict, progress_callback: Callable[[dict], None] | None = None) -> ScanResult:
    def emit(payload: dict) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(payload)
        except Exception:
            pass

    input_folder = Path(config["input_folder"]).resolve()
    output_folder = Path(config["output_folder"]).resolve()
    db_path = Path(config["database_path"]).resolve()
    overwrite_output = bool(config.get("overwrite_output", True))
    link_mode = str(config.get("identity_view_link_mode", "none"))

    emit({"stage": "discovering", "message": "Discovering image files..."})
    discovered_paths = discover_image_paths(
        input_folder,
        exclude_name_contains=config.get("exclude_name_contains"),
    )
    emit(
        {
            "stage": "discovering",
            "message": "Image discovery complete.",
            "discovered_count": len(discovered_paths),
            "total_count": len(discovered_paths),
        }
    )
    scan_id = uuid.uuid4().hex

    conn = open_database(db_path)
    try:
        create_scan(
            conn,
            scan_id=scan_id,
            input_folder=input_folder,
            output_folder=output_folder,
            mode=str(config.get("mode", "realism")),
            discovered_count=len(discovered_paths),
        )

        existing_index = fetch_existing_image_index(conn)
        excluded_tokens = _normalized_excluded_tokens(config.get("exclude_name_contains"))
        if excluded_tokens:
            _blacklist_excluded_existing_records(
                conn=conn,
                existing_index=existing_index,
                input_folder=input_folder,
                excluded_tokens=excluded_tokens,
            )
            existing_index = fetch_existing_image_index(conn)
        to_process: list[Path] = []
        total_discovered = len(discovered_paths)
        for index, path in enumerate(discovered_paths, start=1):
            row = existing_index.get(str(path))
            stat = path.stat()
            if row is None:
                to_process.append(path)
            else:
                if str(row["status"]) == "blacklisted":
                    touch_existing_image(conn, path)
                    continue
                same_size = int(row["size_bytes"]) == int(stat.st_size)
                same_mtime = abs(float(row["mtime"]) - float(stat.st_mtime)) < 0.0001
                if same_size and same_mtime:
                    touch_existing_image(conn, path)
                else:
                    to_process.append(path)

            if index % 25 == 0 or index == total_discovered:
                emit(
                    {
                        "stage": "indexing",
                        "message": "Checking existing image records...",
                        "checked_count": index,
                        "discovered_count": total_discovered,
                        "total_count": total_discovered,
                        "total_to_process": len(to_process),
                    }
                )

        assigned_count = 0
        noise_count = 0
        no_face_count = 0
        processed_count = len(to_process)

        if to_process:
            emit(
                {
                    "stage": "loading",
                    "message": "Loading images for embedding...",
                    "total_to_process": processed_count,
                    "total_count": processed_count,
                    "processed_count": 0,
                }
            )
            loaded_images, unreadable = load_images_from_paths(
                to_process,
                progress_callback=lambda p: emit(
                    {
                        "stage": "loading",
                        "message": "Loading images for embedding...",
                        "processed_count": int(p.get("index", 0)),
                        "total_count": int(p.get("total", processed_count)),
                        "total_to_process": processed_count,
                    }
                ),
            )
            emit(
                {
                    "stage": "embedding",
                    "message": "Extracting face embeddings...",
                    "processed_count": 0,
                    "total_count": len(loaded_images),
                    "total_to_process": processed_count,
                }
            )
            embedder = FaceEmbedder(config)
            embed_result = embedder.extract(
                loaded_images,
                progress_callback=lambda p: emit(
                    {
                        "stage": "embedding",
                        "message": "Extracting face embeddings...",
                        "processed_count": int(p.get("index", 0)),
                        "total_count": int(p.get("total", len(loaded_images))),
                        "embedded_count": int(p.get("embedded_count", 0)),
                        "no_face_count": int(p.get("no_face_count", 0)),
                        "total_to_process": processed_count,
                    }
                ),
            )

            no_face_paths = list(embed_result.no_face_paths) + unreadable
            for no_face_path in no_face_paths:
                _upsert_status(conn, no_face_path, status="no_face", identity_id=None, embedding=None)
            no_face_count += len(no_face_paths)

            if embed_result.embedded_paths:
                assigned_count, noise_count = _assign_and_cluster_new_embeddings(
                    conn=conn,
                    embedded_paths=embed_result.embedded_paths,
                    embeddings=embed_result.embeddings,
                    config=config,
                    progress_callback=lambda p: emit(
                        {
                            **p,
                            "total_to_process": processed_count,
                        }
                    ),
                )

        emit({"stage": "exporting", "message": "Writing metadata and identity views..."})
        rebuild_identity_stats(conn)

        metadata_counts = export_metadata(
            conn,
            output_folder=output_folder,
            input_folder=input_folder,
            path_scope=input_folder,
            overwrite=overwrite_output,
        )
        link_counts = export_identity_views(
            conn,
            output_folder=output_folder,
            link_mode=link_mode,
            path_scope=input_folder,
            overwrite=True,
        )
        identity_count = len(conn.execute("SELECT id FROM identities").fetchall())

        finalize_scan(
            conn,
            scan_id=scan_id,
            processed_count=processed_count,
            assigned_count=assigned_count,
            noise_count=noise_count,
            no_face_count=no_face_count,
        )
        LOGGER.info(
            "Scan %s complete: discovered=%d processed=%d assigned=%d noise=%d no_face=%d identities=%d",
            scan_id,
            len(discovered_paths),
            processed_count,
            assigned_count,
            noise_count,
            no_face_count,
            identity_count,
        )
        emit(
            {
                "stage": "complete",
                "message": "Scan complete.",
                "discovered_count": len(discovered_paths),
                "processed_count": processed_count,
                "assigned_count": assigned_count,
                "noise_count": noise_count,
                "no_face_count": no_face_count,
                "identity_count": identity_count,
                "total_count": processed_count,
            }
        )
        return ScanResult(
            scan_id=scan_id,
            discovered_count=len(discovered_paths),
            processed_count=processed_count,
            assigned_count=assigned_count,
            noise_count=noise_count,
            no_face_count=no_face_count,
            identity_count=identity_count,
            metadata_images=int(metadata_counts["images"]),
            metadata_identities=int(metadata_counts["identities"]),
            linked_images=int(link_counts["linked"]),
        )
    finally:
        conn.close()


def run_lora_export(config: dict) -> dict:
    db_path = Path(config["database_path"]).resolve()
    output_folder = Path(config["output_folder"]).resolve()
    link_mode = str(config.get("lora_link_mode", "hardlink"))
    min_images = int(config.get("lora_min_images", 5))
    face_crop = bool(config.get("lora_face_crop_enabled", False))
    face_crop_size = int(config.get("lora_face_crop_size", 1024))
    face_crop_padding = float(config.get("lora_face_crop_padding", 0.35))
    face_crop_model_name = str(config.get("lora_face_crop_model_name", config.get("model_name", "buffalo_l")))
    face_crop_use_gpu = bool(config.get("lora_face_crop_use_gpu", False))
    identity_ids = config.get("lora_identity_ids")
    selected_ids = None
    if isinstance(identity_ids, list):
        selected_ids = [int(v) for v in identity_ids if int(v) > 0]
    overwrite = bool(config.get("overwrite_output", True))

    conn = open_database(db_path)
    try:
        return export_lora_dataset(
            conn,
            output_folder=output_folder,
            link_mode=link_mode,
            min_images=min_images,
            identity_ids=selected_ids,
            overwrite=overwrite,
            face_crop=face_crop,
            face_crop_size=face_crop_size,
            face_crop_padding=face_crop_padding,
            face_crop_model_name=face_crop_model_name,
            face_crop_use_gpu=face_crop_use_gpu,
        )
    finally:
        conn.close()


def search_identity_candidates(config: dict, image_path: Path, top_k: int = 5) -> list[dict]:
    db_path = Path(config["database_path"]).resolve()
    image_path = image_path.resolve()
    loaded_images, unreadable = load_images_from_paths([image_path])
    if unreadable or not loaded_images:
        return []

    embedder = FaceEmbedder(config)
    result = embedder.extract(loaded_images)
    if result.embeddings.size == 0:
        return []

    query = result.embeddings[0]
    conn = open_database(db_path)
    try:
        identities = fetch_identity_records(conn)
    finally:
        conn.close()
    if not identities:
        return []

    distances = _cosine_distances_to_identities(query, identities)
    rows = sorted(distances, key=lambda item: item["distance"])[: max(1, int(top_k))]
    return rows


def _assign_and_cluster_new_embeddings(
    *,
    conn: sqlite3.Connection,
    embedded_paths: list[Path],
    embeddings: np.ndarray,
    config: dict,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[int, int]:
    def emit(payload: dict) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(payload)
        except Exception:
            pass

    identities = fetch_identity_records(conn)
    assign_eps = resolve_assign_eps(config)
    assigned = 0
    noise = 0

    pending_paths: list[Path] = []
    pending_vectors: list[np.ndarray] = []
    total_embeddings = len(embedded_paths)
    assign_index = 0

    if identities:
        for path, vector in zip(embedded_paths, embeddings):
            assign_index += 1
            best = _closest_identity(vector, identities)
            if best is None or best["distance"] > assign_eps:
                pending_paths.append(path)
                pending_vectors.append(vector)
            else:
                _upsert_status(
                    conn,
                    path,
                    status="assigned",
                    identity_id=int(best["identity_id"]),
                    embedding=vector,
                )
                assigned += 1
            if assign_index % 25 == 0 or assign_index == total_embeddings:
                emit(
                    {
                        "stage": "assigning",
                        "message": "Assigning embeddings to identities...",
                        "processed_count": assign_index,
                        "total_count": total_embeddings,
                        "assigned_count": assigned,
                        "noise_count": noise,
                    }
                )
    else:
        pending_paths = list(embedded_paths)
        pending_vectors = [vector for vector in embeddings]
        assign_index = total_embeddings
        emit(
            {
                "stage": "assigning",
                "message": "No existing identities; all embeddings pending clustering.",
                "processed_count": assign_index,
                "total_count": total_embeddings,
                "assigned_count": assigned,
                "noise_count": noise,
            }
        )

    if pending_vectors:
        emit(
            {
                "stage": "clustering",
                "message": "Clustering unassigned embeddings...",
                "processed_count": 0,
                "total_count": len(pending_vectors),
                "pending_count": len(pending_vectors),
            }
        )
        pending_matrix = np.vstack(pending_vectors).astype(np.float32)
        labels = cluster_faces(pending_matrix, config)

        cluster_ids = sorted({int(v) for v in labels.tolist() if int(v) != -1})
        label_to_identity: dict[int, int] = {}
        for cluster_id in cluster_ids:
            members = pending_matrix[labels == cluster_id]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            label_to_identity[cluster_id] = create_identity(conn, centroid=centroid, label=None)

        clustered_index = 0
        for path, vector, label in zip(pending_paths, pending_matrix, labels.tolist()):
            clustered_index += 1
            if int(label) == -1:
                _upsert_status(conn, path, status="noise", identity_id=None, embedding=vector)
                noise += 1
            else:
                _upsert_status(
                    conn,
                    path,
                    status="assigned",
                    identity_id=label_to_identity[int(label)],
                    embedding=vector,
                )
                assigned += 1
            if clustered_index % 25 == 0 or clustered_index == len(pending_paths):
                emit(
                    {
                        "stage": "assigning",
                        "message": "Finalizing clustered identity assignments...",
                        "processed_count": clustered_index,
                        "total_count": len(pending_paths),
                        "assigned_count": assigned,
                        "noise_count": noise,
                    }
                )

    return assigned, noise


def _upsert_status(
    conn: sqlite3.Connection,
    path: Path,
    *,
    status: str,
    identity_id: int | None,
    embedding: np.ndarray | None,
) -> None:
    stat = path.stat()
    upsert_image(
        conn,
        path=path,
        sha256=sha256_file(path),
        size_bytes=int(stat.st_size),
        mtime=float(stat.st_mtime),
        status=status,
        identity_id=identity_id,
        embedding=embedding,
        face_score=None,
    )


def _closest_identity(vector: np.ndarray, identities: list[IdentityRecord]) -> dict | None:
    if not identities:
        return None
    distances = _cosine_distances_to_identities(vector, identities)
    return min(distances, key=lambda item: item["distance"])


def _cosine_distances_to_identities(vector: np.ndarray, identities: list[IdentityRecord]) -> list[dict]:
    candidates = []
    for identity in identities:
        dot = float(np.dot(vector, identity.centroid))
        dot = max(min(dot, 1.0), -1.0)
        distance = 1.0 - dot
        candidates.append(
            {
                "identity_id": identity.identity_id,
                "label": identity.label,
                "member_count": identity.member_count,
                "distance": distance,
                "similarity": 1.0 - distance,
            }
        )
    return candidates


def _normalized_excluded_tokens(raw_tokens: object) -> list[str]:
    if raw_tokens is None:
        return []
    if isinstance(raw_tokens, str):
        values = [raw_tokens]
    elif isinstance(raw_tokens, (list, tuple, set)):
        values = list(raw_tokens)
    else:
        return []

    tokens: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        if token:
            tokens.append(token)
    return tokens


def _is_path_under_scope(path: Path, scope: Path) -> bool:
    try:
        path.resolve().relative_to(scope.resolve())
        return True
    except ValueError:
        return False


def _name_matches_excluded_tokens(path: Path, excluded_tokens: list[str]) -> bool:
    name_lower = path.name.lower()
    return any(token in name_lower for token in excluded_tokens)


def _blacklist_excluded_existing_records(
    *,
    conn: sqlite3.Connection,
    existing_index: dict[str, sqlite3.Row],
    input_folder: Path,
    excluded_tokens: list[str],
) -> None:
    updated = 0
    for row in existing_index.values():
        candidate = Path(str(row["path"]))
        if not _is_path_under_scope(candidate, input_folder):
            continue
        if not _name_matches_excluded_tokens(candidate, excluded_tokens):
            continue
        status = str(row["status"])
        identity_id = row["identity_id"]
        if status == "blacklisted" and identity_id is None:
            continue
        if update_image_status(
            conn,
            path=candidate,
            status="blacklisted",
            identity_id=None,
            clear_embedding=True,
        ):
            updated += 1
    if updated > 0:
        LOGGER.info("Marked %d previously tracked excluded images as blacklisted.", updated)
