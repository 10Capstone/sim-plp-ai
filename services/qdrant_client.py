from loguru import logger
from qdrant_client import AsyncQdrantClient, models

from core.config import Settings
from core.errors import ServiceUnavailableError


class QdrantClientManager:
    """One asynchronous, read-only client per application lifespan."""

    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None):
        self.settings = settings
        self._client = client
        self.points_count: int | None = None
        self.vector_dimension: int | None = None
        self.connected = False

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise ServiceUnavailableError("Qdrant client is not ready")
        return self._client

    def _validate_collection(self, info: models.CollectionInfo) -> None:
        vectors = info.config.params.vectors
        name = self.settings.QDRANT_VECTOR_NAME
        if isinstance(vectors, dict):
            if name is None or name not in vectors:
                raise ValueError("Collection requires a valid QDRANT_VECTOR_NAME")
            vector = vectors[name]
        else:
            if name is not None or vectors is None:
                raise ValueError("Expected an unnamed dense vector configuration")
            vector = vectors
        if vector.size != self.settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Vector dimension mismatch: collection={vector.size}, "
                f"embedding={self.settings.EMBEDDING_DIMENSION}"
            )
        if vector.distance != models.Distance.COSINE:
            raise ValueError("Collection vector distance must be Cosine")
        self.vector_dimension = vector.size
        self.points_count = info.points_count

    async def startup(self) -> None:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY or None,
                timeout=15,
            )
        try:
            if not await self._client.collection_exists(
                self.settings.QDRANT_COLLECTION_NAME
            ):
                raise ValueError("Configured Qdrant collection does not exist")
            info = await self._client.get_collection(
                self.settings.QDRANT_COLLECTION_NAME
            )
            self._validate_collection(info)
            self.connected = True
        except BaseException:
            await self.close()
            raise
        logger.info(
            "Qdrant ready: collection={}, points={}, dimension={}",
            self.settings.QDRANT_COLLECTION_NAME,
            self.points_count,
            self.vector_dimension,
        )

    async def health(self) -> dict:
        """Check the connection on each probe instead of returning a stale startup flag."""
        try:
            if self._client is None:
                raise ServiceUnavailableError("Qdrant client is closed")
            info = await self._client.get_collection(
                self.settings.QDRANT_COLLECTION_NAME
            )
            self._validate_collection(info)
            self.connected = True
        except Exception as exc:  # noqa: BLE001 - readiness probes must report all dependency failures
            self.connected = False
            logger.warning("Qdrant health check failed ({})", type(exc).__name__)
        return {
            "status": "ok" if self.connected else "degraded",
            "qdrant_status": "connected" if self.connected else "disconnected",
            "collection_name": self.settings.QDRANT_COLLECTION_NAME,
            "points_count": self.points_count if self.connected else None,
            "vector_dimension": self.vector_dimension if self.connected else None,
        }

    async def close(self) -> None:
        self.connected = False
        if self._client is not None:
            await self._client.close()
            self._client = None
