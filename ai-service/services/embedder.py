import asyncio
import math
from typing import Any

from core.config import Settings
from core.latency import track_latency


class QueryEmbedderService:
    def __init__(self, settings: Settings, model: Any = None):
        self.settings = settings
        self._model = model
        self._lock = asyncio.Lock()

    def _encode(self, query: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.EMBEDDING_MODEL_NAME)
        encoded = self._model.encode(
            query,
            normalize_embeddings=True,
            prompt="",
            convert_to_numpy=True,
        )
        vector = [float(value) for value in encoded]
        if len(vector) != self.settings.EMBEDDING_DIMENSION:
            raise ValueError(
                "Embedding dimension does not match the configured collection"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-4):
            raise ValueError("Embedding must be finite and L2 normalized")
        return vector

    async def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        with track_latency("Query Embedding"):
            async with self._lock:
                return await asyncio.to_thread(self._encode, query.strip())
