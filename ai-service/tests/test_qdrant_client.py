from unittest.mock import patch

import pytest
from qdrant_client import models

from core.errors import ServiceUnavailableError
from services.qdrant_client import QdrantClientManager


async def test_startup_reads_existing_collection_and_closes(settings, qdrant_client):
    manager = QdrantClientManager(settings, qdrant_client)
    await manager.startup()
    assert manager.client is qdrant_client
    assert manager.points_count == 677
    assert manager.vector_dimension == 384
    qdrant_client.collection_exists.assert_awaited_once_with(
        settings.QDRANT_COLLECTION_NAME
    )
    qdrant_client.get_collection.assert_awaited_once_with(
        settings.QDRANT_COLLECTION_NAME
    )
    qdrant_client.create_collection.assert_not_called()
    qdrant_client.upsert.assert_not_called()
    await manager.close()
    qdrant_client.close.assert_awaited_once()
    with pytest.raises(ServiceUnavailableError):
        _ = manager.client


async def test_startup_constructs_async_client_from_settings(settings, qdrant_client):
    with patch(
        "services.qdrant_client.AsyncQdrantClient", return_value=qdrant_client
    ) as factory:
        manager = QdrantClientManager(settings)
        await manager.startup()
        factory.assert_called_once_with(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=15,
        )
        await manager.close()


async def test_missing_collection_fails_startup_and_closes(settings, qdrant_client):
    qdrant_client.collection_exists.return_value = False
    manager = QdrantClientManager(settings, qdrant_client)
    with pytest.raises(ValueError, match="does not exist"):
        await manager.startup()
    qdrant_client.close.assert_awaited_once()
    qdrant_client.get_collection.assert_not_awaited()


@pytest.mark.parametrize(
    "size,distance", [(768, models.Distance.COSINE), (384, models.Distance.DOT)]
)
async def test_incompatible_collection_fails_startup(
    settings,
    qdrant_client,
    collection_info,
    size,
    distance,
):
    collection_info.config.params.vectors = models.VectorParams(
        size=size, distance=distance
    )
    manager = QdrantClientManager(settings, qdrant_client)
    with pytest.raises(ValueError):
        await manager.startup()
    qdrant_client.close.assert_awaited_once()


async def test_named_vector_requires_matching_setting(
    settings, qdrant_client, collection_info
):
    collection_info.config.params.vectors = {
        "dense": collection_info.config.params.vectors
    }
    manager = QdrantClientManager(settings, qdrant_client)
    with pytest.raises(ValueError, match="QDRANT_VECTOR_NAME"):
        await manager.startup()

    settings.QDRANT_VECTOR_NAME = "dense"
    manager = QdrantClientManager(settings, qdrant_client)
    await manager.startup()
    assert manager.vector_dimension == 384
    await manager.close()


async def test_health_detects_outage_and_recovery(
    settings, qdrant_client, collection_info
):
    manager = QdrantClientManager(settings, qdrant_client)
    await manager.startup()
    qdrant_client.get_collection.side_effect = ConnectionError("upstream unavailable")
    failed = await manager.health()
    assert failed["status"] == "degraded"
    assert failed["qdrant_status"] == "disconnected"
    assert failed["points_count"] is None
    # A failed probe must not prevent subsequent retrieval attempts from recovering.
    assert manager.client is qdrant_client
    qdrant_client.get_collection.side_effect = None
    collection_info.points_count = 678
    recovered = await manager.health()
    assert recovered["status"] == "ok"
    assert recovered["points_count"] == 678
    await manager.close()
