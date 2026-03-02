import pytest
from rag_forge.config.settings import Settings, VectorStoreConfig
from rag_forge.core.pipeline import RAGPipeline

@pytest.fixture
def settings():
    s = Settings()
    # override teh collection name for the testing purposes
    s.vector_store.collection_name = "test_collection"
    return s

@pytest.fixture
async def pipeline(settings):
    p = RAGPipeline.from_settings(settings)
    yield p
    # delete test collection after each test
    await p._vector_store.delete_collection()

@pytest.mark.asyncio
async def test_ingest_returns_chunk_count(pipeline, tmp_path):
    # create a small temp text file
    f = tmp_path / "test.txt"
    f.write_text("this is a test document. " * 50)
    count = await pipeline.ingest(str(f))
    assert count > 0 

@pytest.mark.asyncio
async def test_query_returns_rag_response(pipeline, tmp_path):
    from rag_forge.generation.response import RAGResponse
    f = tmp_path / "test.txt"
    f.write_text("The capital of Canada is Ottawa. " * 50)
    await pipeline.ingest(str(f))
    response = await pipeline.query("What is the capital of Canada?")
    assert isinstance(response, RAGResponse)
    assert len(response.answer) > 0
    assert response.chunks_used > 0