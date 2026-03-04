import pytest
from unittest.mock import AsyncMock
from rag_forge.cache.embedding_cache import EmbeddingCache
from rag_forge.cache.semantic_cache import SemanticCache
from rag_forge.generation.response import RAGResponse

# embedding cache
@pytest.mark.asyncio
async def test_embedding_cache_returns_same_result():
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_embedder.dimension = 3
    cache = EmbeddingCache(mock_embedder)
    # First call — hits the real embedder
    result1 = await cache.embed(["hello"])
    # Second call — should use cache, not call embedder again
    result2 = await cache.embed(["hello"])
    assert result1 == result2
    assert mock_embedder.embed.call_count == 1  # only called once


@pytest.mark.asyncio
async def test_embedding_cache_splits_cached_and_uncached():
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(side_effect=lambda texts: [[0.1] * 3 for _ in texts])
    mock_embedder.dimension = 3
    cache = EmbeddingCache(mock_embedder)
    await cache.embed(["hello"])          # cache "hello"
    result = await cache.embed(["hello", "world"])  # "hello" cached, "world" not
    assert len(result) == 2
    # "world" required a new embed call
    assert mock_embedder.embed.call_count == 2

# semantic cache
@pytest.mark.asyncio
async def test_semantic_cache_hit():
    mock_embedder = AsyncMock()
    # Return identical embeddings so similarity = 1.0 (above threshold)
    mock_embedder.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    response = RAGResponse(answer="test", citations=[], chunks_used=1,
                           latency_ms=100, model="test")
    cache = SemanticCache(mock_embedder, threshold=0.95)
    await cache.set("What are the terms?", response)
    result = await cache.get("What are the terms?")
    assert result is not None
    assert result.answer == "test"


@pytest.mark.asyncio
async def test_semantic_cache_miss():
    mock_embedder = AsyncMock()
    # Return very different embeddings so similarity is low
    call_count = 0
    async def varying_embed(texts):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [[1.0, 0.0, 0.0]]
        return [[0.0, 1.0, 0.0]]
    mock_embedder.embed = varying_embed
    response = RAGResponse(answer="test", citations=[], chunks_used=1,
                           latency_ms=100, model="test")
    cache = SemanticCache(mock_embedder, threshold=0.95)
    await cache.set("What are the terms?", response)
    result = await cache.get("Something completely different")
    assert result is None

    