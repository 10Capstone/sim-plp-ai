from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.errors import LLMUnavailableError
from services.llm import LLMClientService


async def chunks(*items):
    for item in items:
        yield item


@pytest.mark.parametrize("key_field", ["GEMINI_API_KEY", "LLM_API_KEY"])
async def test_gemini_sdk_client_and_completion(settings, monkeypatch, key_field):
    from google import genai

    setattr(settings, key_field, "test-key")
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=AsyncMock(return_value=SimpleNamespace(text="Jawaban [1]"))
        ),
        aclose=AsyncMock(),
    )
    factory = Mock(return_value=SimpleNamespace(aio=client))
    monkeypatch.setattr(genai, "Client", factory)
    llm = LLMClientService(settings)
    assert factory.call_count == 0
    assert (
        await llm.complete(
            "system",
            [
                {"role": "assistant", "content": "lama"},
                {"role": "user", "content": "baru"},
            ],
        )
        == "Jawaban [1]"
    )
    assert llm.provider == "gemini" and llm.model == settings.LLM_MODEL
    assert factory.call_args.kwargs["api_key"] == "test-key"
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["contents"] == [
        {"role": "model", "parts": [{"text": "lama"}]},
        {"role": "user", "parts": [{"text": "baru"}]},
    ]
    assert kwargs["config"]["system_instruction"] == "system"
    await llm.close()
    await llm.close()
    client.aclose.assert_awaited_once()


async def test_openai_sdk_client_completion_and_cleanup(settings, monkeypatch):
    import openai

    settings.OPENAI_API_KEY = "openai-test"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        ),
        close=AsyncMock(),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    llm = LLMClientService(settings)
    assert await llm.complete("rules", [{"role": "user", "content": "PLP"}]) == "answer"
    assert llm.provider == "openai" and llm.model == settings.OPENAI_MODEL
    assert factory.call_args.kwargs["api_key"] == "openai-test"
    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "PLP"},
    ]
    await llm.close()
    client.close.assert_awaited_once()


def test_explicit_provider_and_model_selection(settings):
    settings.GEMINI_API_KEY = "gemini"
    settings.OPENAI_API_KEY = "openai"
    llm = LLMClientService(settings)
    assert llm.provider == "gemini"
    settings.LLM_MODEL = "custom-openai-model"
    assert llm.provider == "openai" and llm.model == "custom-openai-model"
    settings.LLM_PROVIDER = "gemini"
    assert llm.provider == "gemini"


@pytest.mark.parametrize("provider", ["gemini", "openai"])
async def test_missing_api_key_raises_without_network(settings, provider):
    settings.LLM_PROVIDER = provider
    llm = LLMClientService(settings)
    with pytest.raises(LLMUnavailableError, match="not configured"):
        await llm.complete("rules", [])
    await llm.close()


async def test_gemini_stream_skips_empty_chunks_and_closes(settings):
    stream = chunks(
        SimpleNamespace(text=None),
        SimpleNamespace(text="halo "),
        SimpleNamespace(text="[1]"),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=AsyncMock(return_value=stream))
    )
    llm = LLMClientService(settings, client)
    assert [
        text async for text in llm.stream("rules", [{"role": "user", "content": "q"}])
    ] == ["halo ", "[1]"]
    assert stream.ag_frame is None
    await llm.close()  # Injected clients belong to their caller.


class OpenAIStream:
    def __init__(self):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return chunks(
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="halo"))]
            ),
        )


async def test_openai_stream_skips_usage_and_empty_deltas(settings):
    settings.LLM_PROVIDER = "openai"
    stream = OpenAIStream()
    create = AsyncMock(return_value=stream)
    llm = LLMClientService(
        settings,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    assert [text async for text in llm.stream("rules", [])] == ["halo"]
    assert stream.closed
    assert create.call_args.kwargs["stream"] is True


async def test_cancelling_gemini_stream_closes_upstream(settings):
    closed = []

    async def upstream():
        try:
            yield SimpleNamespace(text="first")
            yield SimpleNamespace(text="second")
        finally:
            closed.append(True)

    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content_stream=AsyncMock(return_value=upstream())
        )
    )
    llm = LLMClientService(settings, client)
    stream = llm.stream("rules", [])
    assert await anext(stream) == "first"
    await stream.aclose()
    assert closed == [True]
