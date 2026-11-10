import json
from contextlib import aclosing
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.v1.retrieve import get_retriever
from core.errors import LLMUnavailableError
from schemas.chat import ChatRequest, ChatResponse, ClearChatResponse
from services.generator import (
    FALLBACK_ANSWER,
    LLM_ERROR_MESSAGE,
    LLMGeneratorService,
    sse_event,
    stream_fallback,
)
from services.memory import ChatMemoryService
from services.query_rewriter import QueryRewriterService
from services.retriever import RetrieverService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def get_memory(request: Request) -> ChatMemoryService:
    return request.app.state.chat_memory


def get_rewriter(request: Request) -> QueryRewriterService:
    return request.app.state.query_rewriter


def get_generator(request: Request) -> LLMGeneratorService:
    return request.app.state.generator


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
    },
)
async def chat(
    body: ChatRequest,
    retriever: Annotated[RetrieverService, Depends(get_retriever)],
    memory: Annotated[ChatMemoryService, Depends(get_memory)],
    rewriter: Annotated[QueryRewriterService, Depends(get_rewriter)],
    generator: Annotated[LLMGeneratorService, Depends(get_generator)],
):
    history = memory.get_history(body.session_id)
    query = (
        await rewriter.rewrite_query(body.message, history) if history else body.message
    )
    context = await retriever.retrieve(
        query=query, **body.retrieval_config.model_dump()
    )
    if not body.stream:
        answer = (
            await generator.generate_answer(query, context, history)
            if context
            else FALLBACK_ANSWER
        )
        memory.add_exchange(body.session_id, body.message, answer)
        return ChatResponse(
            session_id=body.session_id,
            answer=answer,
            rewritten_query=query,
            sources=context,
        )

    async def events():
        parts = []
        stream = (
            generator.stream_answer(query, context, history)
            if context
            else stream_fallback()
        )
        try:
            async with aclosing(stream):
                async for event in stream:
                    if event.startswith("event: token\n"):
                        parts.append(json.loads(event.split("data: ", 1)[1])["token"])
                    elif event.startswith("event: done\n"):
                        memory.add_exchange(
                            body.session_id, body.message, "".join(parts)
                        )
                    yield event
        except LLMUnavailableError:
            # HTTP headers have already been sent: signal failure within SSE.
            yield sse_event("error", {"status": "failed", "message": LLM_ERROR_MESSAGE})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/{session_id}", response_model=ClearChatResponse)
async def clear_chat(
    session_id: str, memory: Annotated[ChatMemoryService, Depends(get_memory)]
):
    memory.clear(session_id)
    return ClearChatResponse(session_id=session_id)
