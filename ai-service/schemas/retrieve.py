from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from core.filters import build_query_filter


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.5, ge=-1, le=1)
    filters: dict[str, Any] | None = None
    use_rerank: bool = Field(
        default=False,
        validation_alias=AliasChoices("use_rerank", "use_reranker"),
    )
    top_k_initial: int = Field(default=15, ge=1, le=1000)
    top_k_final: int | None = Field(default=None, ge=1, le=100)

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        build_query_filter(value)
        return value

    @model_validator(mode="after")
    def validate_rerank_limits(self) -> "RetrievalConfig":
        if self.top_k_final is not None and not self.use_rerank:
            raise ValueError("top_k_final requires use_rerank=true")
        if self.use_rerank and self.top_k_initial < (self.top_k_final or self.top_k):
            raise ValueError("top_k_initial must be at least the final result limit")
        return self


class RetrieveRequest(RetrievalConfig):
    query: str = Field(min_length=1, max_length=10000)


class RetrievedChunk(BaseModel):
    point_id: str
    score: float
    text: str
    metadata: dict[str, Any]
    rerank_score: float | None = None


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    total: int


class HealthResponse(BaseModel):
    status: str
    qdrant_status: str
    collection_name: str
    points_count: int | None
    vector_dimension: int | None
