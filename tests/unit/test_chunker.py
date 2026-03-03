import pytest
from unittest.mock import AsyncMock
from rag_forge.document.models import Document
from rag_forge.document.chunker import RecursiveChunker, HybridChunker, SemanticChunker
from rag_forge.config.settings import ChunkingConfig

@pytest.fixture
def config():
    # Return a ChunkingConfig with small chunk_size for predictable test behaviour
    return ChunkingConfig(chunk_size=100, chunk_overlap=10)

@pytest.fixture
def chunker(config):
    return RecursiveChunker(config)

@pytest.fixture
def sample_document():
    # Return a Document with enough content to produce multiple chunks
    return Document(
        content="word " * 200,
        source="test.txt",
        file_type="txt"
    )


# Recursive Chunker
@pytest.mark.asyncio
async def test_count_greater_than_zero(chunker, sample_document):
    # chunk count > 0
    chunks = await chunker.chunk(sample_document)
    assert len(chunks) > 0
        

@pytest.mark.asyncio
async def test_metadata_propagated(chunker, sample_document):
    chunks = await chunker.chunk(sample_document)
    for chunk in chunks:
        assert chunk.source == sample_document.source

@pytest.mark.asyncio
async def test_chunk_index_sequential(chunker, sample_document):
    chunks = await chunker.chunk(sample_document)
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))

@pytest.mark.asyncio
async def test_no_empty_chunks(chunker, sample_document):
    chunks = await chunker.chunk(sample_document)
    assert all(chunk.content.strip() != "" for chunk in chunks)


# Semantic Chunker
@pytest.mark.asyncio
async def test_semantic_returns_chunks(sample_document):
    # mock embedder that returns random embeddings
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(
        side_effect=lambda texts: [[0.1] * 384 for _ in texts]
    )
    chunker = SemanticChunker(mock_embedder, ChunkingConfig(semantic_threshold=0.9))
    chunks = await chunker.chunk(sample_document)
    assert len(chunks) > 0

# Hybrid Chunker
@pytest.mark.asyncio
async def test_hybrid_children_have_parent(sample_document):
    # Children mus thave parent_id, parent_id must exist in get_parent()
    chunker = HybridChunker(ChunkingConfig(parent_chunk_size=200, child_chunk_size=50))
    children = await chunker.chunk(sample_document)
    for child in children:
        assert child.parent_id is not None
        assert chunker.get_parent(child.parent_id) is not None

@pytest.mark.asyncio
async def test_hybrid_children_smaller_than_parent(sample_document):
    # Every child chunk must be shorted that its parent
    chunker = HybridChunker(ChunkingConfig(parent_chunk_size=200, child_chunk_size=50))
    children = await chunker.chunk(sample_document)
    for child in children:
        parent = chunker.get_parent(child.parent_id)
        assert len(child.content) <= len(parent.content)

