import asyncio
import math
from typing import Any

from loguru import logger

from core.config import Settings
from core.errors import ServiceUnavailableError
from core.filters import build_query_filter
from core.latency import track_latency
from services.embedder import QueryEmbedderService
from services.qdrant_client import QdrantClientManager


class RetrieverService:
    def __init__(
        self,
        settings: Settings,
        manager: QdrantClientManager,
        embedder: QueryEmbedderService,
        ranker: Any = None,
    ):
        self.settings = settings
        self.manager = manager
        self.embedder = embedder
        self._ranker = ranker
        self._ranker_lock = asyncio.Lock()

    def _rerank(self, query: str, chunks: list[dict], limit: int) -> list[dict]:
        from flashrank import Ranker, RerankRequest

        if self._ranker is None:
            # Multilingual model, including Indonesian. Downloaded only when requested.
            self._ranker = Ranker(model_name="ms-marco-MultiBERT-L-12")
        passages = [{"id": i, "text": chunk["text"]} for i, chunk in enumerate(chunks)]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            {**chunks[int(item["id"])], "rerank_score": float(item["score"])}
            for item in ranked[:limit]
        ]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filters: dict | None = None,
        *,
        use_rerank: bool = False,
        top_k_initial: int = 15,
        top_k_final: int | None = None,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must be a non-empty string")
        final_limit = top_k if top_k_final is None else top_k_final
        if top_k < 1 or final_limit < 1 or not -1 <= score_threshold <= 1:
            raise ValueError("Invalid retrieval limits or cosine score threshold")
        if use_rerank and top_k_initial < final_limit:
            raise ValueError("top_k_initial must be at least the final result limit")
        query_filter = build_query_filter(filters)
        try:
            vector = await self.embedder.embed_query(query)
            with track_latency("Qdrant Retrieval"):
                response = await self.manager.client.query_points(
                    collection_name=self.settings.QDRANT_COLLECTION_NAME,
                    query=vector,
                    using=self.settings.QDRANT_VECTOR_NAME,
                    query_filter=query_filter,
                    limit=top_k_initial if use_rerank else top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            chunks = []
            for point in response.points:
                if not math.isfinite(point.score) or point.score < score_threshold:
                    continue
                payload = dict(point.payload or {})
                text = payload.pop(self.settings.PAYLOAD_TEXT_KEY, None)
                if not isinstance(text, str) or not text.strip():
                    logger.warning(
                        "Skipping Qdrant point {} without chunk text", point.id
                    )
                    continue
                chunks.append(
                    {
                        "point_id": str(point.id),
                        "score": float(point.score),
                        "text": text,
                        "metadata": payload,
                    }
                )
            if use_rerank and chunks:
                with track_latency("Reranking"):
                    async with self._ranker_lock:
                        return await asyncio.to_thread(
                            self._rerank, query, chunks, final_limit
                        )
            return chunks
        except Exception as exc:
            # Do not expose upstream URLs, keys, or response bodies to HTTP clients.
            logger.error("Retrieval dependency failed ({})", type(exc).__name__)
            raise ServiceUnavailableError(
                "Retrieval service is temporarily unavailable"
            ) from exc
