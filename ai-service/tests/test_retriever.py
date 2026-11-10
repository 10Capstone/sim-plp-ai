from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client.http import models

from core.errors import ServiceUnavailableError
from services.retriever import RetrieverService


@pytest.fixture
def retrieval(settings, qdrant_client):
    manager = SimpleNamespace(client=qdrant_client)
    embedder = AsyncMock()
    embedder.embed_query.return_value = [1.0] + [0.0] * 383
    ranker = Mock()
    service = RetrieverService(settings, manager, embedder, ranker)
    return service, embedder, ranker


def point(point_id=1, text="Isi panduan", **metadata):
    return models.ScoredPoint(
        id=point_id,
        version=0,
        score=0.8,
        payload={"text": text, **metadata},
    )


async def test_query_points_unnamed_vector_and_metadata(
    retrieval, qdrant_client, settings
):
    service, embedder, ranker = retrieval
    metadata = {
        "chunk_id": "c1",
        "doc_index": 0,
        "page": 4,
        "source_document": "Panduan.pdf",
        "chapter": "I",
        "section": "Prosedur",
        "config_id": "cs96_ov32",
        "extra": "preserved",
    }
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[point(**metadata)]
    )
    results = await service.retrieve(
        "prosedur PLP", filters={"chapter": "I", "page": 4}
    )
    assert results == [
        {
            "point_id": "1",
            "score": 0.8,
            "text": "Isi panduan",
            "metadata": metadata,
        }
    ]
    kwargs = qdrant_client.query_points.call_args.kwargs
    assert kwargs["query"] == embedder.embed_query.return_value
    assert kwargs["collection_name"] == settings.QDRANT_COLLECTION_NAME
    assert kwargs["using"] is None
    assert kwargs["limit"] == 5
    assert kwargs["score_threshold"] == 0.5
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False
    assert kwargs["query_filter"].must[0].match.value == "I"
    ranker.rerank.assert_not_called()


async def test_custom_text_key_and_named_vector(retrieval, qdrant_client):
    service, _, _ = retrieval
    service.settings.PAYLOAD_TEXT_KEY = "content"
    service.settings.QDRANT_VECTOR_NAME = "dense"
    item = point()
    item.payload = {"content": "Custom chunk", "chunk_id": "c1"}
    qdrant_client.query_points.return_value = models.QueryResponse(points=[item])
    result = await service.retrieve("PLP")
    assert result[0]["text"] == "Custom chunk"
    assert result[0]["metadata"] == {"chunk_id": "c1"}
    assert qdrant_client.query_points.call_args.kwargs["using"] == "dense"


async def test_reranker_overfetches_reorders_and_keeps_original_scores(
    retrieval, qdrant_client
):
    service, _, ranker = retrieval
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[
            point(1, "Pertama"),
            point(2, "Kedua"),
            point(3, "Ketiga"),
        ]
    )
    ranker.rerank.return_value = [
        {"id": 2, "score": 0.98},
        {"id": 0, "score": 0.7},
        {"id": 1, "score": 0.1},
    ]
    results = await service.retrieve(
        "PLP", use_rerank=True, top_k_initial=15, top_k_final=2
    )
    assert qdrant_client.query_points.call_args.kwargs["limit"] == 15
    assert [item["point_id"] for item in results] == ["3", "1"]
    assert results[0]["score"] == 0.8
    assert results[0]["rerank_score"] == 0.98
    request = ranker.rerank.call_args.args[0]
    assert request.query == "PLP"
    assert len(request.passages) == 3


async def test_reranker_defaults_final_limit_to_top_k(retrieval, qdrant_client):
    service, _, ranker = retrieval
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[point(1), point(2)]
    )
    ranker.rerank.return_value = [{"id": 1, "score": 0.9}, {"id": 0, "score": 0.5}]
    results = await service.retrieve("PLP", top_k=1, use_rerank=True)
    assert len(results) == 1
    assert results[0]["point_id"] == "2"


async def test_empty_and_missing_text_results_skip_reranker(retrieval, qdrant_client):
    service, _, ranker = retrieval
    assert await service.retrieve("PLP", use_rerank=True) == []
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[point(text=None)]
    )
    assert await service.retrieve("PLP", use_rerank=True) == []
    ranker.rerank.assert_not_called()


async def test_low_and_nonfinite_scores_are_filtered_before_reranking(
    retrieval, qdrant_client
):
    service, _, ranker = retrieval
    low, invalid, boundary = point(1), point(2), point(3)
    low.score, invalid.score, boundary.score = 0.64, float("nan"), 0.65
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[low, invalid]
    )
    assert await service.retrieve("PLP", score_threshold=0.65, use_rerank=True) == []
    ranker.rerank.assert_not_called()
    qdrant_client.query_points.return_value = models.QueryResponse(
        points=[low, invalid, boundary]
    )
    assert [
        chunk["point_id"]
        for chunk in await service.retrieve("PLP", score_threshold=0.65)
    ] == ["3"]


@pytest.mark.parametrize("dependency", ["qdrant", "embedder", "ranker"])
async def test_dependency_errors_are_sanitized(retrieval, qdrant_client, dependency):
    service, embedder, ranker = retrieval
    error = RuntimeError("private upstream response")
    if dependency == "qdrant":
        qdrant_client.query_points.side_effect = error
    elif dependency == "embedder":
        embedder.embed_query.side_effect = error
    else:
        qdrant_client.query_points.return_value = models.QueryResponse(points=[point()])
        ranker.rerank.side_effect = error
    with pytest.raises(ServiceUnavailableError, match="temporarily unavailable") as exc:
        await service.retrieve("PLP", use_rerank=True)
    assert "private" not in str(exc.value)
