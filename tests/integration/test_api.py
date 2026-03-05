import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch
from rag_forge.api.app import create_app
from rag_forge.generation.response import RAGResponse

@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.query = AsyncMock()
    pipeline.ingest = AsyncMock()
    pipeline._llm = MagicMock()
    pipeline._llm.stream = AsyncMock()
    return pipeline


@pytest.fixture
def app(mock_pipeline):
    application = create_app()
    application.state.pipeline = mock_pipeline
    return application


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_query_returns_response(client, mock_pipeline):
    mock_response = RAGResponse(answer="test answer", citations=[],
                                chunks_used=1, latency_ms=100, model="test")
    mock_pipeline.query = AsyncMock(return_value=mock_response)
    response = await client.post("/query", json={"question": "What are the terms?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "test answer"


@pytest.mark.asyncio
async def test_ingest_accepted(client, mock_pipeline):
    mock_pipeline.ingest = AsyncMock(return_value=10)
    response = await client.post("/ingest", json={"path": "data/raw/sample.pdf"})
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_query_stream_returns_sse(client, mock_pipeline):
    async def mock_stream(prompt):
        for token in ["Hello", " world"]:
            yield token

    mock_pipeline._llm.stream = mock_stream
    response = await client.post("/query/stream", json={"question": "test?"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]