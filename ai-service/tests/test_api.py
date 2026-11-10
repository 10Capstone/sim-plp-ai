from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main
from api.v1.retrieve import get_retriever
from core.errors import ServiceUnavailableError
from services.qdrant_client import QdrantClientManager


@pytest.fixture
def api(settings, qdrant_client, monkeypatch):
    manager = QdrantClientManager(settings, qdrant_client)
    monkeypatch.setattr(main, "QdrantClientManager", lambda config: manager)
    app = main.create_app(settings)
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    app.dependency_overrides[get_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield client, retriever, qdrant_client
    qdrant_client.close.assert_awaited_once()


def test_health_returns_current_collection_info(api):
    client, _, _ = api
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "qdrant_status": "connected",
        "collection_name": "plp_indoe5_cs96_ov32",
        "points_count": 677,
        "vector_dimension": 384,
    }


def test_health_outage_returns_503_without_stale_point_count(api):
    client, _, qdrant_client = api
    qdrant_client.get_collection.side_effect = ConnectionError(
        "private upstream response"
    )
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["qdrant_status"] == "disconnected"
    assert response.json()["points_count"] is None
    assert "private" not in response.text


def test_retrieve_endpoint_response_and_arguments(api):
    client, retriever, _ = api
    retriever.retrieve.return_value = [
        {
            "point_id": "42",
            "score": 0.8,
            "text": "Panduan PLP",
            "metadata": {"page": 3, "chunk_id": "c42"},
            "rerank_score": 0.95,
        }
    ]
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "  prosedur PLP  ",
            "top_k": 2,
            "score_threshold": 0.6,
            "filters": {"chapter": "I"},
            "use_rerank": True,
            "top_k_initial": 15,
            "top_k_final": 1,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "prosedur PLP"
    assert result["total"] == 1
    assert result["results"][0]["metadata"]["page"] == 3
    retriever.retrieve.assert_awaited_once_with(
        query="prosedur PLP",
        top_k=2,
        score_threshold=0.6,
        filters={"chapter": "I"},
        use_rerank=True,
        top_k_initial=15,
        top_k_final=1,
    )


def test_empty_retrieval_is_successful(api):
    client, _, _ = api
    response = client.post("/api/v1/retrieve", json={"query": "PLP"})
    assert response.status_code == 200
    assert response.json() == {"query": "PLP", "results": [], "total": 0}


def test_use_reranker_alias(api):
    client, retriever, _ = api
    response = client.post(
        "/api/v1/retrieve", json={"query": "PLP", "use_reranker": True}
    )
    assert response.status_code == 200
    assert retriever.retrieve.call_args.kwargs["use_rerank"] is True


@pytest.mark.parametrize(
    "invalid",
    [
        {"query": " "},
        {"top_k": 0},
        {"top_k": 101},
        {"score_threshold": 1.1},
        {"score_threshold": -1.1},
        {"top_k_final": 2},
        {"use_rerank": True, "top_k_initial": 2, "top_k_final": 3},
        {"filters": {"page": None}},
        {"filters": {"page": {"invalid": 3}}},
        {"filters": {"must": [{"bad_condition": True}]}},
        {"unknown_parameter": 1},
    ],
)
def test_invalid_requests_return_422_before_retrieval(api, invalid):
    client, retriever, _ = api
    response = client.post("/api/v1/retrieve", json={"query": "PLP", **invalid})
    assert response.status_code == 422
    retriever.retrieve.assert_not_awaited()


def test_retrieval_dependency_failure_returns_503(api):
    client, retriever, _ = api
    retriever.retrieve.side_effect = ServiceUnavailableError(
        "private upstream response"
    )
    response = client.post("/api/v1/retrieve", json={"query": "PLP"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Retrieval service is temporarily unavailable"}


def test_openapi_exposes_endpoints(api):
    client, _, _ = api
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/api/v1/retrieve" in schema["paths"]


def test_startup_failure_prevents_serving_and_closes_client(
    settings, qdrant_client, monkeypatch
):
    qdrant_client.collection_exists.return_value = False
    manager = QdrantClientManager(settings, qdrant_client)
    monkeypatch.setattr(main, "QdrantClientManager", lambda config: manager)
    with (
        pytest.raises(ValueError, match="does not exist"),
        TestClient(main.create_app(settings)),
    ):
        pass
    qdrant_client.close.assert_awaited_once()
