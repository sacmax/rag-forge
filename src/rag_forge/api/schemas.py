from pydantic import BaseModel

class IngestRequest(BaseModel):
    path: str # local file path or url

class IngestResponse(BaseModel):
    chunks: int  # number of chunks stored
    path: str

class QueryRequest(BaseModel):
    question: str 
    k: int = 5  # number of chunks to retrieve

class CitationSchema(BaseModel):
    source: str
    page: int | None
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    chunks_used: int
    latency_ms: float
    model: str

class HealthResponse(BaseModel):
    status: str
    version: str