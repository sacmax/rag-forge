from pydantic import BaseModel

class Citation(BaseModel):
    source: str
    page: int | None
    chunk_id: str

class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chunks_used: int
    latency_ms: float
    model: str
    