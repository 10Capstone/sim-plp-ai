from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

from core.config import Settings
from core.errors import LLMUnavailableError


class LLMClientService:
    """Lazy async provider adapter shared by rewriting and answer generation."""

    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    @property
    def provider(self) -> str:
        if self.settings.LLM_PROVIDER != "auto":
            return self.settings.LLM_PROVIDER
        if self.settings.LLM_MODEL.startswith("gemini") and (
            self.settings.GEMINI_API_KEY or self.settings.LLM_API_KEY
        ):
            return "gemini"
        return "openai" if self.settings.OPENAI_API_KEY else "gemini"

    @property
    def model(self) -> str:
        if self.provider == "openai" and self.settings.LLM_MODEL.startswith("gemini"):
            return self.settings.OPENAI_MODEL
        return self.settings.LLM_MODEL

    def _get_client(self):
        if self._client is None:
            if self.provider == "gemini":
                from google import genai

                key = self.settings.GEMINI_API_KEY or self.settings.LLM_API_KEY
                if not key:
                    raise LLMUnavailableError("LLM API key is not configured")
                self._client = genai.Client(
                    api_key=key,
                    http_options={
                        "timeout": int(self.settings.LLM_TIMEOUT_SECONDS * 1000)
                    },
                ).aio
            else:
                from openai import AsyncOpenAI

                if not self.settings.OPENAI_API_KEY:
                    raise LLMUnavailableError("LLM API key is not configured")
                self._client = AsyncOpenAI(
                    api_key=self.settings.OPENAI_API_KEY,
                    timeout=self.settings.LLM_TIMEOUT_SECONDS,
                    max_retries=0,
                )
        return self._client

    @staticmethod
    def _gemini_messages(messages: list[dict]) -> list[dict]:
        return [
            {
                "role": "model" if item["role"] == "assistant" else "user",
                "parts": [{"text": item["content"]}],
            }
            for item in messages
        ]

    async def complete(self, system: str, messages: list[dict]) -> str:
        client = self._get_client()
        if self.provider == "gemini":
            response = await client.models.generate_content(
                model=self.model,
                contents=self._gemini_messages(messages),
                config={"system_instruction": system, "temperature": 0},
            )
            return response.text or ""
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        system: str,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        if self.provider == "gemini":
            stream = await client.models.generate_content_stream(
                model=self.model,
                contents=self._gemini_messages(messages),
                config={"system_instruction": system, "temperature": 0},
            )
            async with aclosing(stream):
                async for chunk in stream:
                    if chunk.text:
                        yield chunk.text
        else:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                stream=True,
            )
            async with stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            if self.provider == "gemini":
                await self._client.aclose()
            else:
                await self._client.close()
            self._client = None
