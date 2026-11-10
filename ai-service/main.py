from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from api.v1.chat import router as chat_router
from api.v1.retrieve import router as retrieve_router
from core.config import Settings, get_settings
from core.errors import LLMUnavailableError, ServiceUnavailableError
from schemas.retrieve import HealthResponse
from services.embedder import QueryEmbedderService
from services.generator import LLM_ERROR_MESSAGE, LLMGeneratorService
from services.llm import LLMClientService
from services.memory import ChatMemoryService
from services.qdrant_client import QdrantClientManager
from services.query_rewriter import QueryRewriterService
from services.retriever import RetrieverService


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings if settings is not None else get_settings()
        manager = QdrantClientManager(config)
        llm = LLMClientService(config)
        try:
            await manager.startup()
            app.state.qdrant_manager = manager
            app.state.retriever = RetrieverService(
                config, manager, QueryEmbedderService(config)
            )
            app.state.chat_memory = ChatMemoryService()
            app.state.query_rewriter = QueryRewriterService(config, llm)
            app.state.generator = LLMGeneratorService(config, llm)
            yield
        finally:
            try:
                await llm.close()
            finally:
                await manager.close()

    app = FastAPI(
        title="Qdrant Chatbot API Service", version="0.1.0", lifespan=lifespan
    )
    app.include_router(retrieve_router)
    app.include_router(chat_router)

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable(request: Request, exc: LLMUnavailableError):
        return JSONResponse(status_code=503, content={"detail": LLM_ERROR_MESSAGE})

    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable(request: Request, exc: ServiceUnavailableError):
        return JSONResponse(
            status_code=503,
            content={"detail": "Retrieval service is temporarily unavailable"},
        )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health(request: Request, response: Response):
        result = await request.app.state.qdrant_manager.health()
        if result["status"] != "ok":
            response.status_code = 503
        return result

    return app


app = create_app()
