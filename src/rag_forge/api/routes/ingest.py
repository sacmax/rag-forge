from fastapi import APIRouter, BackgroundTasks, Request
from rag_forge.api.schemas import IngestRequest, IngestResponse

router = APIRouter()

async def _do_ingest(pipeline, path: str) -> None:
    await pipeline.ingest(path)

@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(body: IngestRequest, background_tasks: BackgroundTasks, request: Request):
    pipeline = request.app.state.pipeline
    background_tasks.add_task(_do_ingest, pipeline, body.path)
    return IngestResponse(chunks=0, path=body.path)
