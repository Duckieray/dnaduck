from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IdentityRecord:
    identity_id: int
    label: str | None
    member_count: int
    centroid: np.ndarray
    created_at: str
    updated_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def open_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            member_count INTEGER NOT NULL DEFAULT 0,
            centroid BLOB,
            centroid_dim INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime REAL NOT NULL,
            status TEXT NOT NULL,
            identity_id INTEGER REFERENCES identities(id) ON DELETE SET NULL,
            embedding BLOB,
            embedding_dim INTEGER,
            face_score REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_images_identity_id ON images(identity_id);
        CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);

        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            input_folder TEXT NOT NULL,
            output_folder TEXT NOT NULL,
            mode TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            discovered_count INTEGER NOT NULL DEFAULT 0,
            processed_count INTEGER NOT NULL DEFAULT 0,
            assigned_count INTEGER NOT NULL DEFAULT 0,
            noise_count INTEGER NOT NULL DEFAULT 0,
            no_face_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()


def serialize_embedding(vector: np.ndarray | None) -> tuple[bytes | None, int | None]:
    if vector is None:
        return None, None
    value = np.asarray(vector, dtype=np.float32).ravel()
    return value.tobytes(), int(value.shape[0])


def deserialize_embedding(blob: bytes | None, dim: int | None) -> np.ndarray | None:
    if blob is None or dim is None:
        return None
    vector = np.frombuffer(blob, dtype=np.float32)
    if dim != vector.shape[0]:
        return None
    return vector


def fetch_existing_image_index(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT id, path, sha256, size_bytes, mtime, status, identity_id
        FROM images
        """
    ).fetchall()
    return {str(row["path"]): row for row in rows}


def create_scan(
    conn: sqlite3.Connection,
    scan_id: str,
    input_folder: Path,
    output_folder: Path,
    mode: str,
    discovered_count: int,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT OR REPLACE INTO scans (
            id, input_folder, output_folder, mode, started_at, discovered_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (scan_id, str(input_folder), str(output_folder), mode, now, discovered_count),
    )
    conn.commit()


def finalize_scan(
    conn: sqlite3.Connection,
    scan_id: str,
    *,
    processed_count: int,
    assigned_count: int,
    noise_count: int,
    no_face_count: int,
) -> None:
    conn.execute(
        """
        UPDATE scans
        SET finished_at = ?, processed_count = ?, assigned_count = ?, noise_count = ?, no_face_count = ?
        WHERE id = ?
        """,
        (utc_now_iso(), processed_count, assigned_count, noise_count, no_face_count, scan_id),
    )
    conn.commit()


def create_identity(conn: sqlite3.Connection, centroid: np.ndarray, label: str | None = None) -> int:
    blob, dim = serialize_embedding(centroid)
    now = utc_now_iso()
    cursor = conn.execute(
        """
        INSERT INTO identities (label, member_count, centroid, centroid_dim, created_at, updated_at)
        VALUES (?, 0, ?, ?, ?, ?)
        """,
        (label, blob, dim, now, now),
    )
    conn.commit()
    return int(cursor.lastrowid)


def upsert_image(
    conn: sqlite3.Connection,
    *,
    path: Path,
    sha256: str,
    size_bytes: int,
    mtime: float,
    status: str,
    identity_id: int | None,
    embedding: np.ndarray | None,
    face_score: float | None = None,
) -> int:
    now = utc_now_iso()
    blob, dim = serialize_embedding(embedding)
    existing = conn.execute("SELECT id FROM images WHERE path = ?", (str(path),)).fetchone()
    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO images (
                path, sha256, size_bytes, mtime, status, identity_id,
                embedding, embedding_dim, face_score, created_at, updated_at, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(path),
                sha256,
                int(size_bytes),
                float(mtime),
                status,
                identity_id,
                blob,
                dim,
                face_score,
                now,
                now,
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    image_id = int(existing["id"])
    conn.execute(
        """
        UPDATE images
        SET sha256 = ?, size_bytes = ?, mtime = ?, status = ?, identity_id = ?,
            embedding = ?, embedding_dim = ?, face_score = ?, updated_at = ?, last_seen_at = ?
        WHERE id = ?
        """,
        (
            sha256,
            int(size_bytes),
            float(mtime),
            status,
            identity_id,
            blob,
            dim,
            face_score,
            now,
            now,
            image_id,
        ),
    )
    conn.commit()
    return image_id


def touch_existing_image(conn: sqlite3.Connection, path: Path) -> None:
    conn.execute(
        "UPDATE images SET last_seen_at = ?, updated_at = ? WHERE path = ?",
        (utc_now_iso(), utc_now_iso(), str(path)),
    )
    conn.commit()


def fetch_identity_records(conn: sqlite3.Connection) -> list[IdentityRecord]:
    rows = conn.execute(
        """
        SELECT id, label, member_count, centroid, centroid_dim, created_at, updated_at
        FROM identities
        ORDER BY id ASC
        """
    ).fetchall()
    result: list[IdentityRecord] = []
    for row in rows:
        centroid = deserialize_embedding(row["centroid"], row["centroid_dim"])
        if centroid is None:
            continue
        result.append(
            IdentityRecord(
                identity_id=int(row["id"]),
                label=row["label"],
                member_count=int(row["member_count"]),
                centroid=centroid,
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        )
    return result


def rebuild_identity_stats(conn: sqlite3.Connection) -> None:
    identity_rows = conn.execute("SELECT id FROM identities ORDER BY id ASC").fetchall()
    for row in identity_rows:
        identity_id = int(row["id"])
        emb_rows = conn.execute(
            """
            SELECT embedding, embedding_dim
            FROM images
            WHERE identity_id = ? AND status = 'assigned' AND embedding IS NOT NULL
            """,
            (identity_id,),
        ).fetchall()
        vectors = [
            deserialize_embedding(emb_row["embedding"], emb_row["embedding_dim"])
            for emb_row in emb_rows
        ]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
            continue

        matrix = np.vstack(vectors).astype(np.float32)
        centroid = matrix.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        blob, dim = serialize_embedding(centroid)
        conn.execute(
            """
            UPDATE identities
            SET member_count = ?, centroid = ?, centroid_dim = ?, updated_at = ?
            WHERE id = ?
            """,
            (int(matrix.shape[0]), blob, dim, utc_now_iso(), identity_id),
        )
    conn.commit()


def list_identities(conn: sqlite3.Connection, min_members: int = 1) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, label, member_count, created_at, updated_at
        FROM identities
        WHERE member_count >= ?
        ORDER BY member_count DESC, id ASC
        """,
        (int(min_members),),
    ).fetchall()


def list_identity_images(conn: sqlite3.Connection, identity_id: int) -> list[sqlite3.Row]:
    return list_identity_images_page(conn, identity_id=identity_id, limit=None, offset=0)


def count_identity_images(conn: sqlite3.Connection, identity_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(1) AS c FROM images WHERE identity_id = ?",
        (int(identity_id),),
    ).fetchone()
    return int(row["c"]) if row is not None else 0


def list_identity_images_page(
    conn: sqlite3.Connection,
    *,
    identity_id: int,
    limit: int | None = 120,
    offset: int = 0,
) -> list[sqlite3.Row]:
    safe_offset = max(0, int(offset))
    if limit is None:
        return conn.execute(
            """
            SELECT id, path, status, created_at, updated_at
            FROM images
            WHERE identity_id = ?
            ORDER BY path ASC
            """,
            (int(identity_id),),
        ).fetchall()

    safe_limit = max(1, int(limit))
    return conn.execute(
        """
        SELECT id, path, status, created_at, updated_at
        FROM images
        WHERE identity_id = ?
        ORDER BY path ASC
        LIMIT ? OFFSET ?
        """,
        (int(identity_id), safe_limit, safe_offset),
    ).fetchall()


def compute_identity_drift(conn: sqlite3.Connection, identity_id: int) -> float | None:
    row = conn.execute(
        "SELECT centroid, centroid_dim FROM identities WHERE id = ?",
        (int(identity_id),),
    ).fetchone()
    if row is None:
        return None
    centroid = deserialize_embedding(row["centroid"], row["centroid_dim"])
    if centroid is None:
        return None
    emb_rows = conn.execute(
        """
        SELECT embedding, embedding_dim
        FROM images
        WHERE identity_id = ? AND status = 'assigned' AND embedding IS NOT NULL
        """,
        (int(identity_id),),
    ).fetchall()
    vectors = [
        deserialize_embedding(emb_row["embedding"], emb_row["embedding_dim"])
        for emb_row in emb_rows
    ]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    matrix = np.vstack(vectors).astype(np.float32)
    sims = np.clip(matrix @ centroid.reshape(-1, 1), -1.0, 1.0).reshape(-1)
    distances = 1.0 - sims
    return float(np.mean(distances))


def list_all_images(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, path, sha256, status, identity_id, size_bytes, mtime, created_at, updated_at, last_seen_at
        FROM images
        ORDER BY path ASC
        """
    ).fetchall()


def fetch_image_by_path(conn: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, path, status, identity_id, sha256, size_bytes, mtime, created_at, updated_at, last_seen_at
        FROM images
        WHERE path = ?
        """,
        (str(path),),
    ).fetchone()


def update_image_status(
    conn: sqlite3.Connection,
    *,
    path: Path,
    status: str,
    identity_id: int | None,
    clear_embedding: bool = False,
) -> bool:
    existing = conn.execute("SELECT id FROM images WHERE path = ?", (str(path),)).fetchone()
    if existing is None:
        return False

    now = utc_now_iso()
    if clear_embedding:
        conn.execute(
            """
            UPDATE images
            SET status = ?, identity_id = ?, embedding = NULL, embedding_dim = NULL, updated_at = ?
            WHERE path = ?
            """,
            (str(status), identity_id, now, str(path)),
        )
    else:
        conn.execute(
            """
            UPDATE images
            SET status = ?, identity_id = ?, updated_at = ?
            WHERE path = ?
            """,
            (str(status), identity_id, now, str(path)),
        )
    conn.commit()
    return True


def update_identity_label(conn: sqlite3.Connection, identity_id: int, label: str | None) -> bool:
    cursor = conn.execute(
        "UPDATE identities SET label = ?, updated_at = ? WHERE id = ?",
        (label, utc_now_iso(), int(identity_id)),
    )
    conn.commit()
    return cursor.rowcount > 0


def merge_identities(conn: sqlite3.Connection, target_id: int, source_ids: list[int]) -> None:
    source_ids = sorted({int(v) for v in source_ids if int(v) != int(target_id)})
    if not source_ids:
        return

    placeholders = ",".join("?" for _ in source_ids)
    conn.execute(
        f"UPDATE images SET identity_id = ?, updated_at = ? WHERE identity_id IN ({placeholders})",
        (int(target_id), utc_now_iso(), *source_ids),
    )
    conn.execute(f"DELETE FROM identities WHERE id IN ({placeholders})", tuple(source_ids))
    conn.commit()
    rebuild_identity_stats(conn)


def fetch_image_with_embedding(conn: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, path, status, identity_id, embedding, embedding_dim
        FROM images
        WHERE path = ?
        """,
        (str(path),),
    ).fetchone()
