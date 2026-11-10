import asyncio
from unittest.mock import AsyncMock

import pytest

from services.query_rewriter import QueryRewriterService

HISTORY = [
    {"role": "user", "content": "Apa syarat PLP?", "timestamp": "now"},
    {"role": "assistant", "content": "Syarat PLP ...", "timestamp": "now"},
]


async def test_first_turn_does_not_call_llm(settings):
    llm = AsyncMock()
    service = QueryRewriterService(settings, llm)
    assert await service.rewrite_query("Apa syarat PLP?", []) == "Apa syarat PLP?"
    llm.complete.assert_not_awaited()


async def test_followup_uses_history_without_timestamps(settings):
    llm = AsyncMock()
    llm.complete.return_value = "  Berapa biaya mengikuti PLP ITK?  "
    service = QueryRewriterService(settings, llm)
    assert (
        await service.rewrite_query("Berapa biayanya?", HISTORY)
        == "Berapa biaya mengikuti PLP ITK?"
    )
    prompt, messages = llm.complete.call_args.args
    assert "HANYA query" in prompt
    assert messages[-1] == {"role": "user", "content": "Berapa biayanya?"}
    assert messages[0] == {"role": "user", "content": "Apa syarat PLP?"}


@pytest.mark.parametrize("result", ["", "   ", "a" * 10001])
async def test_unusable_output_falls_back(settings, result):
    llm = AsyncMock()
    llm.complete.return_value = result
    assert (
        await QueryRewriterService(settings, llm).rewrite_query("asli", HISTORY)
        == "asli"
    )


@pytest.mark.parametrize("error", [RuntimeError("private-key"), TimeoutError()])
async def test_provider_failure_falls_back(settings, error):
    llm = AsyncMock()
    llm.complete.side_effect = error
    assert (
        await QueryRewriterService(settings, llm).rewrite_query("asli", HISTORY)
        == "asli"
    )


async def test_real_unconfigured_provider_falls_back(settings):
    assert await QueryRewriterService(settings).rewrite_query("asli", HISTORY) == "asli"


async def test_rewrite_timeout_and_cancellation(settings):
    settings.LLM_TIMEOUT_SECONDS = 0.01
    llm = AsyncMock()

    async def slow(*args):
        await asyncio.Event().wait()

    llm.complete.side_effect = slow
    assert (
        await QueryRewriterService(settings, llm).rewrite_query("asli", HISTORY)
        == "asli"
    )
    llm.complete.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await QueryRewriterService(settings, llm).rewrite_query("asli", HISTORY)
