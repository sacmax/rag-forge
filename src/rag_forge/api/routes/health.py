from fastapi import APIRouter
from rag_forge.api.schemas import HealthResponse


router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.5.0")