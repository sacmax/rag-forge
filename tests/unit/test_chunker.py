import pytest
from rag_forge.document.models import Document
from rag_forge.document.chunker import RecursiveChunker
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

def test_count_greater_than_zero(chunker, sample_document):
    # chunk count > 0
    chunks = chunker.chunk(sample_document)
    assert len(chunks) > 0
        


def test_metadata_propagated(chunker, sample_document):
    chunks = chunker.chunk(sample_document)
    for chunk in chunks:
        assert chunk.source == sample_document.source

def test_chunk_index_sequential(chunker, sample_document):
    chunks = chunker.chunk(sample_document)
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))

def test_no_empty_chunks(chunker, sample_document):
    chunks = chunker.chunk(sample_document)
    assert all(chunk.content.strip() != "" for chunk in chunks)



