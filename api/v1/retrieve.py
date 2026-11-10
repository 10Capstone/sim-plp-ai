from typing import Annotated

from fastapi import APIRouter, Depends, Request

from schemas.retrieve import RetrieveRequest, RetrieveResponse
from services.retriever import RetrieverService

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


def get_retriever(request: Request) -> RetrieverService:
    return request.app.state.retriever


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    service: Annotated[RetrieverService, Depends(get_retriever)],
) -> RetrieveResponse:
    results = await service.retrieve(**body.model_dump())
    return RetrieveResponse(query=body.query, results=results, total=len(results))
