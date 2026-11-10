from pydantic import BaseModel, ConfigDict, Field

from schemas.retrieve import RetrievalConfig, RetrievedChunk


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10000)
    stream: bool = True
    retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    sources: list[RetrievedChunk]


class ClearChatResponse(BaseModel):
    session_id: str
    status: str = "cleared"
