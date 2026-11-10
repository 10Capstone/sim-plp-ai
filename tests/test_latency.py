import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from qdrant_client.http import models

from core.errors import ServiceUnavailableError
from services.embedder import QueryEmbedderService
from services.generator import LLMGeneratorService
from services.retriever import RetrieverService
from tests.test_generator import CONTEXT, token_stream


@pytest.fixture
def latency_records():
    records = []
    sink = logger.add(lambda message: records.append(message.record))
    yield records
    logger.remove(sink)


async def test_all_pipeline_stages_log_numeric_milliseconds(
    settings, qdrant_client, latency_records
):
    embedder = QueryEmbedderService(
        settings, Mock(encode=Mock(return_value=[1.0] + [0.0] * 383))
    )
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[
            models.ScoredPoint(id=42, version=0, score=0.9, payload={"text": "PLP"})
        ]
    )
    ranker = Mock()
    ranker.rerank.return_value = [{"id": 0, "score": 0.99}]
    retriever = RetrieverService(
        settings, SimpleNamespace(client=qdrant_client), embedder, ranker
    )
    context = await retriever.retrieve("PLP", use_rerank=True)
    llm = SimpleNamespace(
        complete=AsyncMock(return_value="Jawaban [1]"),
        stream=lambda *args: token_stream("Jawaban [1]"),
    )
    generator = LLMGeneratorService(settings, llm)
    await generator.generate_answer("q", context, [])
    _ = [event async for event in generator.stream_answer("q", CONTEXT, [])]
    timed = [record for record in latency_records if "latency_ms" in record["extra"]]
    assert [record["extra"]["stage"] for record in timed] == [
        "Query Embedding",
        "Qdrant Retrieval",
        "Reranking",
        "LLM Generation",
        "LLM Generation",
    ]
    for record in timed:
        assert math.isfinite(record["extra"]["latency_ms"])
        assert record["extra"]["latency_ms"] >= 0
        assert " ms" in record["message"]


async def test_failed_retrieval_still_logs_latency(
    settings, qdrant_client, latency_records
):
    qdrant_client.query_points.side_effect = ConnectionError("secret-key")
    retriever = RetrieverService(
        settings,
        SimpleNamespace(client=qdrant_client),
        SimpleNamespace(embed_query=AsyncMock(return_value=[1.0])),
    )
    with pytest.raises(ServiceUnavailableError):
        await retriever.retrieve("PLP")
    assert any(
        record["extra"].get("stage") == "Qdrant Retrieval" for record in latency_records
    )
    assert all("secret-key" not in record["message"] for record in latency_records)
