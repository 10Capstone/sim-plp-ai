from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    QDRANT_URL: str = Field(min_length=1)
    QDRANT_API_KEY: str = Field(default="", repr=False)
    QDRANT_COLLECTION_NAME: str = "plp_indoe5_cs96_ov32"
    QDRANT_VECTOR_NAME: str | None = None
    PAYLOAD_TEXT_KEY: str = "text"
    EMBEDDING_MODEL_NAME: str = "LazarusNLP/all-indo-e5-small-v4"
    EMBEDDING_DIMENSION: int = Field(default=384, gt=0)
    LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_API_KEY: str = Field(default="", repr=False)
    OPENAI_API_KEY: str = Field(default="", repr=False)
    OPENAI_BASE_URL: str | None = None
    LLM_API_KEY: str = Field(default="", repr=False)
    LLM_PROVIDER: Literal["auto", "gemini", "openai"] = "auto"
    OPENAI_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = Field(default=30, gt=0)

    @field_validator("QDRANT_VECTOR_NAME", mode="before")
    @classmethod
    def empty_vector_name(cls, value: str | None) -> str | None:
        return value.strip() or None if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
