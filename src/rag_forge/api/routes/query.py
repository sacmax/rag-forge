from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from rag_forge.api.schemas import QueryRequest, QueryResponse
import json

router = APIRouter()

async def _stream_tokens(pipeline, question: str, k: int):
    """Generator that yields Server-Sent Events(SSE) formatted tokens"""
    async for token in pipeline._llm.stream(question):
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/query")
async def query(body: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline
    #non streaming path - returns full query response
    response = await pipeline.query(body.question, k=body.k)
    return response

@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline

    #streaming path- return SSE stream of tokens
    return StreamingResponse(
        _stream_tokens(pipeline, body.question, body.k),
        media_type="text/event-stream"
    )