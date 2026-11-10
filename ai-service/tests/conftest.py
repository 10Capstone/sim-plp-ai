from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from qdrant_client.http import models

from core.config import Settings


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        QDRANT_URL="http://localhost:6333",
        QDRANT_API_KEY="test",
        GEMINI_API_KEY="",
        OPENAI_API_KEY="",
        LLM_API_KEY="",
    )


@pytest.fixture
def collection_info():
    return SimpleNamespace(
        points_count=677,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        ),
    )


@pytest.fixture
def qdrant_client(collection_info):
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info
    client.query_points.return_value = models.QueryResponse(points=[])
    return client
