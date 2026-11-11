import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing

from loguru import logger

from core.config import Settings
from core.errors import LLMUnavailableError
from core.latency import track_latency
from services.llm import LLMClientService

FALLBACK_ANSWER = (
    "Maaf, informasi tersebut tidak ditemukan dalam panduan PLP yang tersedia."
)
LLM_ERROR_MESSAGE = (
    "Layanan pembuat jawaban sementara tidak tersedia. Silakan coba lagi."
)
SYSTEM_PROMPT = f"""Anda adalah Asisten Panduan PLP / MBKM ITK.
Jawab pertanyaan dalam bahasa Indonesia secara to-the-point, ramah, dan HANYA berdasarkan dokumen panduan yang diberikan. Setiap klaim faktual wajib disertai sitasi [1], [2], dan seterusnya sesuai nomor sumber.

Aturan:
1. Langsung jawab ke inti tanpa basa-basi atau kalimat pembuka klise (jangan gunakan "Mungkin maksud Anda...").
2. Jika pengguna memakai istilah umum atau singkatan (seperti berkas, syarat, stupen), jawab berdasarkan fakta terkait yang ada di panduan (seperti proposal atau syarat kegiatan).
3. Jangan pernah menyebut istilah teknis seperti "konteks", "potongan dokumen", atau "prompt".
4. Jika panduan sama sekali tidak memuat informasi yang berkaitan dengan pertanyaan pengguna, jawab persis:
   "{FALLBACK_ANSWER}"."""


def sse_event(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def source_references(context: list[dict]) -> list[dict]:
    return [
        {key: chunk[key] for key in ("point_id", "score", "metadata")}
        for chunk in context
    ]


async def stream_fallback() -> AsyncGenerator[str, None]:
    yield sse_event("sources", [])
    yield sse_event("token", {"token": FALLBACK_ANSWER})
    yield sse_event("done", {"status": "completed"})


class LLMGeneratorService:
    def __init__(self, settings: Settings, llm: LLMClientService | None = None):
        self.settings = settings
        self.llm = llm if llm is not None else LLMClientService(settings)

    @staticmethod
    def _messages(query: str, context: list[dict], history: list[dict]) -> list[dict]:
        # JSON escaping keeps source text and user input clearly delimited as data.
        sources = [
            {"citation": f"[{i}]", "text": chunk["text"], "metadata": chunk["metadata"]}
            for i, chunk in enumerate(context, 1)
        ]
        messages = [
            {"role": item["role"], "content": item["content"]} for item in history[-6:]
        ]
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"konteks_dokumen": sources, "pertanyaan": query},
                    ensure_ascii=False,
                ),
            }
        )
        return messages

    async def generate_answer(
        self, query: str, context: list[dict], history: list[dict]
    ) -> str:
        if not context:
            return FALLBACK_ANSWER
        try:
            with track_latency("LLM Generation"):
                async with asyncio.timeout(self.settings.LLM_TIMEOUT_SECONDS):
                    answer = await self.llm.complete(
                        SYSTEM_PROMPT, self._messages(query, context, history)
                    )
            return answer.strip() or FALLBACK_ANSWER
        except Exception as exc:
            logger.error("LLM generation failed ({})", type(exc).__name__)
            raise LLMUnavailableError(LLM_ERROR_MESSAGE) from exc

    async def stream_answer(
        self,
        query: str,
        context: list[dict],
        history: list[dict],
    ) -> AsyncGenerator[str, None]:
        if not context:
            async for event in stream_fallback():
                yield event
            return
        yield sse_event("sources", source_references(context))
        try:
            with track_latency("LLM Generation"):
                async with asyncio.timeout(self.settings.LLM_TIMEOUT_SECONDS):
                    has_text = False
                    async with aclosing(
                        self.llm.stream(
                            SYSTEM_PROMPT, self._messages(query, context, history)
                        )
                    ) as stream:
                        async for token in stream:
                            if token:
                                has_text = has_text or bool(token.strip())
                                yield sse_event("token", {"token": token})
                    if not has_text:
                        yield sse_event("token", {"token": FALLBACK_ANSWER})
        except Exception as exc:
            logger.error("LLM streaming failed ({})", type(exc).__name__)
            raise LLMUnavailableError(LLM_ERROR_MESSAGE) from exc
        yield sse_event("done", {"status": "completed"})
