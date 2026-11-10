import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.errors import LLMUnavailableError
from services.generator import FALLBACK_ANSWER, SYSTEM_PROMPT, LLMGeneratorService

CONTEXT = [
    {
        "point_id": "42",
        "score": 0.9,
        "text": "PLP wajib 20 SKS.",
        "metadata": {"page": 3, "title": "Panduan PLP"},
    },
    {
        "point_id": "43",
        "score": 0.8,
        "text": "Pendaftaran melalui prodi.",
        "metadata": {"page": 4},
    },
]


def parse_event(frame):
    event, data = frame.strip().split("\n", 1)
    return event.removeprefix("event: "), json.loads(data.removeprefix("data: "))


async def token_stream(*tokens):
    for token in tokens:
        yield token


async def test_grounded_prompt_numbered_context_and_history(settings):
    llm = AsyncMock()
    llm.complete.return_value = "  PLP wajib 20 SKS [1].  "
    generator = LLMGeneratorService(settings, llm)
    history = [{"role": "assistant", "content": "sebelumnya", "timestamp": "now"}]
    assert (
        await generator.generate_answer("syarat?", CONTEXT, history)
        == "PLP wajib 20 SKS [1]."
    )
    system, messages = llm.complete.call_args.args
    assert system == SYSTEM_PROMPT
    assert FALLBACK_ANSWER in system and "HANYA" in system
    assert messages[0] == {"role": "assistant", "content": "sebelumnya"}
    payload = json.loads(messages[-1]["content"])
    assert payload["pertanyaan"] == "syarat?"
    assert [item["citation"] for item in payload["konteks_dokumen"]] == ["[1]", "[2]"]
    assert payload["konteks_dokumen"][0]["text"] == CONTEXT[0]["text"]


async def test_empty_context_never_calls_llm_in_either_mode(settings):
    llm = AsyncMock()
    generator = LLMGeneratorService(settings, llm)
    assert await generator.generate_answer("q", [], []) == FALLBACK_ANSWER
    events = [
        parse_event(frame) async for frame in generator.stream_answer("q", [], [])
    ]
    assert events == [
        ("sources", []),
        ("token", {"token": FALLBACK_ANSWER}),
        ("done", {"status": "completed"}),
    ]
    llm.complete.assert_not_awaited()
    llm.stream.assert_not_called()


async def test_exact_sse_framing_and_preservation_of_unicode_newlines(settings):
    tokens = ['Prosedur "PLP" ', "baris\nberikut — [1]"]
    llm = SimpleNamespace(stream=Mock(side_effect=lambda *args: token_stream(*tokens)))
    generator = LLMGeneratorService(settings, llm)
    frames = [frame async for frame in generator.stream_answer("q", CONTEXT, [])]
    assert all(frame.endswith("\n\n") for frame in frames)
    events = [parse_event(frame) for frame in frames]
    assert [event for event, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1] == [
        {key: chunk[key] for key in ("point_id", "score", "metadata")}
        for chunk in CONTEXT
    ]
    assert "".join(
        data["token"] for event, data in events if event == "token"
    ) == "".join(tokens)
    assert events[-1][1] == {"status": "completed"}


async def test_blank_llm_answer_uses_fallback(settings):
    llm = SimpleNamespace(
        complete=AsyncMock(return_value="   "),
        stream=Mock(side_effect=lambda *args: token_stream("")),
    )
    generator = LLMGeneratorService(settings, llm)
    assert await generator.generate_answer("q", CONTEXT, []) == FALLBACK_ANSWER
    events = [
        parse_event(frame) async for frame in generator.stream_answer("q", CONTEXT, [])
    ]
    assert events[1] == ("token", {"token": FALLBACK_ANSWER})


async def test_llm_errors_are_sanitized(settings):
    llm = AsyncMock()
    llm.complete.side_effect = RuntimeError("secret-key")
    with pytest.raises(LLMUnavailableError) as exc:
        await LLMGeneratorService(settings, llm).generate_answer("q", CONTEXT, [])
    assert "secret" not in str(exc.value)


async def test_partial_stream_failure_never_emits_done(settings):
    async def failing(*args):
        yield "partial"
        raise RuntimeError("secret-key")

    generator = LLMGeneratorService(settings, SimpleNamespace(stream=failing))
    events = []
    with pytest.raises(LLMUnavailableError):
        async for frame in generator.stream_answer("q", CONTEXT, []):
            events.append(parse_event(frame))
    assert [event for event, _ in events] == ["sources", "token"]


async def test_generator_timeout(settings):
    settings.LLM_TIMEOUT_SECONDS = 0.01

    async def slow(*args):
        await asyncio.Event().wait()

    with pytest.raises(LLMUnavailableError):
        await LLMGeneratorService(
            settings, SimpleNamespace(complete=slow)
        ).generate_answer("q", CONTEXT, [])


async def test_stream_timeout_does_not_emit_done(settings):
    settings.LLM_TIMEOUT_SECONDS = 0.01

    async def slow(*args):
        yield "partial"
        await asyncio.Event().wait()

    events = []
    with pytest.raises(LLMUnavailableError):
        async for frame in LLMGeneratorService(
            settings, SimpleNamespace(stream=slow)
        ).stream_answer("q", CONTEXT, []):
            events.append(parse_event(frame))
    assert [event for event, _ in events] == ["sources", "token"]


async def test_cancelled_stream_closes_upstream(settings):
    closed = []

    async def upstream(*args):
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    stream = LLMGeneratorService(
        settings, SimpleNamespace(stream=upstream)
    ).stream_answer("q", CONTEXT, [])
    assert parse_event(await anext(stream))[0] == "sources"
    assert parse_event(await anext(stream)) == ("token", {"token": "first"})
    await stream.aclose()
    assert closed == [True]
