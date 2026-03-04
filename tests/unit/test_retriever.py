import pytest
import numpy as np
from unittest.mock import AsyncMock
from rag_forge.retrieval.hybrid import reciprocal_rank_fusion
from rag_forge.retrieval.retriever import mmr
from rag_forge.retrieval.query_transform import HyDETransformer, MultiQueryTransformer
from rag_forge.document.models import Chunk


def make_chunk(id: str, content: str) -> Chunk:
    return Chunk(id=id, content=content, document_id="doc1",
                 chunk_index=0, source="test.txt")

# RRF
def test_rrf_merges_two_lists():
    # chunk appearing in both lists should rank higher than one appearing in one
    list1 = ["a", "b", "c"]
    list2 = ["b", "c", "a"]
    result = reciprocal_rank_fusion([list1, list2])
    assert result[0] == "b"  # "b" is rank 2 in list1 and rank 1 in list2


def test_rrf_single_list_preserves_order():
    ids = ["x", "y", "z"]
    result = reciprocal_rank_fusion([ids])
    assert result == ids

# MMR
def test_mmr_returns_k_chunks():
    query_emb = [1.0, 0.0]
    chunks = [make_chunk(str(i), f"chunk {i}") for i in range(5)]
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5], [0.8, 0.2]]
    result = mmr(query_emb, chunks, embeddings, k=3)
    assert len(result) == 3


def test_mmr_no_duplicates():
    query_emb = [1.0, 0.0]
    chunks = [make_chunk(str(i), f"chunk {i}") for i in range(4)]
    embeddings = [[1.0, 0.0]] * 4  # all identical — MMR should still not duplicate
    result = mmr(query_emb, chunks, embeddings, k=3)
    ids = [c.id for c in result]
    assert len(ids) == len(set(ids))

# HyDe
@pytest.mark.asyncio
async def test_hyde_returns_string():
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value="A hypothetical passage about payment terms.")
    transformer = HyDETransformer(mock_llm)
    result = await transformer.transform("What are the payment terms?")
    assert isinstance(result, str) and len(result) > 0

# MultiQuery
@pytest.mark.asyncio
async def test_multi_query_includes_original():
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value="Paraphrase 1\nParaphrase 2\nParaphrase 3")
    transformer = MultiQueryTransformer(mock_llm, n=3)
    queries = await transformer.transform("original question")
    assert queries[0] == "original question"  # original is always first
    assert len(queries) >= 2