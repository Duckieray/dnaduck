from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import LoadedImage, get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: np.ndarray
    embedded_paths: list[Path]
    no_face_paths: list[Path]


class FaceEmbedder:
    """InsightFace-powered embedding extractor for one-face-per-image batches."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = str(config.get("model_name", "buffalo_l"))
        self.use_gpu = bool(config.get("use_gpu", True))
        self.det_size = tuple(config.get("det_size", (640, 640)))
        if len(self.det_size) != 2:
            raise ValueError("det_size must be a 2-item tuple/list")

        self.app = self._build_face_app()

    def _build_face_app(self):
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "InsightFace is required but unavailable. Install dependencies from requirements.txt."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(name=self.model_name, providers=providers)
        # ctx_id=0 targets primary GPU; -1 targets CPU.
        ctx_id = 0 if self.use_gpu else -1
        app.prepare(ctx_id=ctx_id, det_size=tuple(int(v) for v in self.det_size))
        LOGGER.info(
            "InsightFace initialized (model=%s, gpu=%s, providers=%s)",
            self.model_name,
            self.use_gpu,
            ",".join(providers),
        )
        return app

    def extract(self, images: Iterable[LoadedImage], progress_callback=None) -> EmbeddingResult:
        embeddings: list[np.ndarray] = []
        embedded_paths: list[Path] = []
        no_face_paths: list[Path] = []

        ordered_images = list(images)
        total = len(ordered_images)
        for index, item in enumerate(ordered_images, start=1):
            faces = self.app.get(item.array_bgr)
            if not faces:
                no_face_paths.append(item.path)
            else:
                face = self._pick_primary_face(faces)
                vector = self._normalize_embedding(face)
                embeddings.append(vector)
                embedded_paths.append(item.path)

            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "index": index,
                            "total": total,
                            "path": str(item.path),
                            "embedded_count": len(embedded_paths),
                            "no_face_count": len(no_face_paths),
                        }
                    )
                except Exception:
                    pass

        if embeddings:
            emb_array = np.vstack(embeddings).astype(np.float32)
        else:
            emb_array = np.empty((0, 0), dtype=np.float32)

        LOGGER.info(
            "Embedding complete: %d embedded, %d no-face",
            len(embedded_paths),
            len(no_face_paths),
        )
        return EmbeddingResult(
            embeddings=emb_array,
            embedded_paths=embedded_paths,
            no_face_paths=no_face_paths,
        )

    @staticmethod
    def _pick_primary_face(faces):
        # Phase A assumes one face/image; if more exist, choose the largest bbox.
        def face_area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

        return max(faces, key=face_area)

    @staticmethod
    def _normalize_embedding(face) -> np.ndarray:
        vector = getattr(face, "normed_embedding", None)
        if vector is None:
            vector = getattr(face, "embedding", None)
            if vector is None:
                raise RuntimeError("Face object did not include an embedding vector.")
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise RuntimeError("Encountered zero-norm face embedding.")
            vector = vector / norm
        return np.asarray(vector, dtype=np.float32)
