from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import main
from api.v1.chat import chat
from api.v1.retrieve import get_retriever
from core.errors import ServiceUnavailableError
from schemas.chat import ChatRequest
from services.generator import FALLBACK_ANSWER, LLM_ERROR_MESSAGE, LLMGeneratorService
from services.memory import ChatMemoryService
from services.qdrant_client import QdrantClientManager
from tests.test_generator import CONTEXT, parse_event, token_stream


@pytest.fixture
def chat_api(settings, qdrant_client, monkeypatch):
    llm = SimpleNamespace(
        complete=AsyncMock(return_value="PLP wajib 20 SKS [1]."),
        stream=Mock(
            side_effect=lambda *args: token_stream("PLP wajib ", "20 SKS [1].")
        ),
        close=AsyncMock(),
    )
    manager = QdrantClientManager(settings, qdrant_client)
    monkeypatch.setattr(main, "QdrantClientManager", lambda config: manager)
    monkeypatch.setattr(main, "LLMClientService", lambda config: llm)
    app = main.create_app(settings)
    retriever = AsyncMock()
    retriever.retrieve.return_value = CONTEXT
    app.dependency_overrides[get_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            app=app,
            retriever=retriever,
            llm=llm,
            memory=app.state.chat_memory,
        )
    llm.close.assert_awaited_once()
    qdrant_client.close.assert_awaited_once()


def request(api, **kwargs):
    return api.client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "message": "Syarat PLP?", "stream": False, **kwargs},
    )


def decode_sse(response):
    return [
        parse_event(frame + "\n\n") for frame in response.text.split("\n\n") if frame
    ]


def test_json_contract_sources_citations_and_history(chat_api):
    response = request(chat_api)
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "s1"
    assert payload["answer"] == "PLP wajib 20 SKS [1]."
    assert payload["rewritten_query"] == "Syarat PLP?"
    assert payload["sources"][0]["text"] == CONTEXT[0]["text"]
    assert payload["sources"][0]["metadata"] == CONTEXT[0]["metadata"]
    assert [(m["role"], m["content"]) for m in chat_api.memory.get_history("s1")] == [
        ("user", "Syarat PLP?"),
        ("assistant", payload["answer"]),
    ]
    chat_api.llm.complete.assert_awaited_once()


def test_followup_rewrite_is_used_for_retrieval_and_original_message_is_saved(chat_api):
    request(chat_api)
    chat_api.llm.complete.side_effect = ["Berapa biaya PLP ITK?", "Biaya PLP [1]."]
    response = request(chat_api, message="Biayanya?")
    assert response.json()["rewritten_query"] == "Berapa biaya PLP ITK?"
    assert (
        chat_api.retriever.retrieve.call_args.kwargs["query"] == "Berapa biaya PLP ITK?"
    )
    assert chat_api.memory.get_history("s1")[-2]["content"] == "Biayanya?"
    prompt, messages = chat_api.llm.complete.call_args.args
    assert messages[0] == {"role": "user", "content": "Syarat PLP?"}
    assert "HANYA" in prompt


def test_rewrite_failure_does_not_break_chat(chat_api):
    request(chat_api)
    chat_api.llm.complete.side_effect = [RuntimeError("private"), "Jawaban [1]"]
    response = request(chat_api, message="Biayanya?")
    assert response.status_code == 200
    assert response.json()["rewritten_query"] == "Biayanya?"


def test_retrieval_configuration_and_alias(chat_api):
    response = request(
        chat_api,
        retrieval_config={
            "top_k": 2,
            "score_threshold": 0.65,
            "use_reranker": True,
            "top_k_initial": 20,
            "top_k_final": 1,
            "filters": {"page": 3},
        },
    )
    assert response.status_code == 200
    chat_api.retriever.retrieve.assert_awaited_once_with(
        query="Syarat PLP?",
        top_k=2,
        score_threshold=0.65,
        use_rerank=True,
        top_k_initial=20,
        top_k_final=1,
        filters={"page": 3},
    )


@pytest.mark.parametrize("stream", [False, True])
def test_no_context_bypasses_generator_and_llm(chat_api, monkeypatch, stream):
    chat_api.retriever.retrieve.return_value = []
    generator = chat_api.app.state.generator
    generate = Mock(side_effect=AssertionError("Generator must not be called"))
    monkeypatch.setattr(generator, "generate_answer", generate)
    monkeypatch.setattr(generator, "stream_answer", generate)
    response = request(chat_api, stream=stream)
    assert response.status_code == 200
    if stream:
        assert decode_sse(response) == [
            ("sources", []),
            ("token", {"token": FALLBACK_ANSWER}),
            ("done", {"status": "completed"}),
        ]
    else:
        assert response.json()["answer"] == FALLBACK_ANSWER
        assert response.json()["sources"] == []
    chat_api.llm.complete.assert_not_awaited()
    chat_api.llm.stream.assert_not_called()
    assert chat_api.memory.get_history("s1")[-1]["content"] == FALLBACK_ANSWER


def test_default_stream_sse_contract_and_memory(chat_api):
    response = chat_api.client.post(
        "/api/v1/chat", json={"session_id": "s1", "message": "Syarat PLP?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = decode_sse(response)
    assert [event for event, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1][0] == {
        key: CONTEXT[0][key] for key in ("point_id", "score", "metadata")
    }
    assert events[-1][1] == {"status": "completed"}
    assert chat_api.memory.get_history("s1")[-1]["content"] == "PLP wajib 20 SKS [1]."
    chat_api.llm.complete.assert_not_awaited()


def test_stream_failure_is_sanitized_and_does_not_save_partial_answer(chat_api):
    async def fail(*args):
        yield "partial"
        raise ConnectionError("private API key")

    chat_api.llm.stream.side_effect = fail
    response = request(chat_api, stream=True)
    events = decode_sse(response)
    assert [event for event, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"status": "failed", "message": LLM_ERROR_MESSAGE}
    assert "private" not in response.text
    assert chat_api.memory.get_history("s1") == []


def test_json_llm_failure_returns_503_and_preserves_history(chat_api):
    chat_api.llm.complete.side_effect = ConnectionError("private API key")
    response = request(chat_api)
    assert response.status_code == 503
    assert response.json() == {"detail": LLM_ERROR_MESSAGE}
    assert chat_api.memory.get_history("s1") == []


@pytest.mark.parametrize("stream", [False, True])
def test_retrieval_failure_is_503_before_stream_headers(chat_api, stream):
    chat_api.retriever.retrieve.side_effect = ServiceUnavailableError("private")
    response = request(chat_api, stream=stream)
    assert response.status_code == 503
    assert "private" not in response.text
    assert chat_api.memory.get_history("s1") == []
    chat_api.llm.complete.assert_not_awaited()
    chat_api.llm.stream.assert_not_called()


def test_delete_is_idempotent_and_preserves_other_sessions(chat_api):
    request(chat_api)
    request(chat_api, session_id="s2")
    assert chat_api.client.delete("/api/v1/chat/s1").json() == {
        "session_id": "s1",
        "status": "cleared",
    }
    assert chat_api.client.delete("/api/v1/chat/s1").status_code == 200
    assert chat_api.memory.get_history("s1") == []
    assert len(chat_api.memory.get_history("s2")) == 2
    chat_api.llm.complete.reset_mock()
    request(chat_api)
    chat_api.llm.complete.assert_awaited_once()  # No rewrite after deletion.


@pytest.mark.parametrize(
    "invalid",
    [
        {"session_id": " "},
        {"session_id": "a" * 201},
        {"session_id": None},
        {"message": " "},
        {"message": "a" * 10001},
        {"message": 3},
        {"unknown": True},
        {"stream": "invalid"},
        {"retrieval_config": None},
        {"retrieval_config": {"top_k": 0}},
        {"retrieval_config": {"score_threshold": 1.1}},
        {"retrieval_config": {"score_threshold": "inf"}},
        {"retrieval_config": {"filters": {"page": None}}},
        {
            "retrieval_config": {
                "use_rerank": True,
                "top_k_initial": 1,
                "top_k_final": 2,
            }
        },
        {"retrieval_config": {"top_k_final": 1}},
        {"retrieval_config": {"query": "override"}},
    ],
)
def test_invalid_chat_never_calls_dependencies(chat_api, invalid):
    response = request(chat_api, **invalid)
    assert response.status_code == 422
    chat_api.retriever.retrieve.assert_not_awaited()
    chat_api.llm.complete.assert_not_awaited()


def test_missing_required_fields(chat_api):
    assert chat_api.client.post("/api/v1/chat", json={}).status_code == 422


def test_openapi_exposes_json_sse_and_delete(chat_api):
    paths = chat_api.client.get("/openapi.json").json()["paths"]
    content = paths["/api/v1/chat"]["post"]["responses"]["200"]["content"]
    assert "application/json" in content and "text/event-stream" in content
    assert "delete" in paths["/api/v1/chat/{session_id}"]


async def test_disconnect_closes_generation_without_saving_partial_turn(settings):
    closed = []

    async def upstream(*args):
        try:
            yield "partial"
            yield "rest"
        finally:
            closed.append(True)

    memory = ChatMemoryService()
    retriever = AsyncMock()
    retriever.retrieve.return_value = CONTEXT
    response = await chat(
        ChatRequest(session_id="s", message="q"),
        retriever,
        memory,
        AsyncMock(),
        LLMGeneratorService(settings, SimpleNamespace(stream=upstream)),
    )
    assert parse_event(await anext(response.body_iterator))[0] == "sources"
    assert parse_event(await anext(response.body_iterator))[0] == "token"
    await response.body_iterator.aclose()
    assert closed == [True]
    assert memory.get_history("s") == []
