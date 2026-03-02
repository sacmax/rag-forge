from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_forge.core.pipeline import RAGPipeline

async def evaluate_retrieval(
        pipeline: "RAGPipeline",
        golden_qa: list[dict],
        k: int = 5,
) -> dict[str, float]:
    """
    Args:
        golden_qa: list of {"question": str, "relevant_chunk_ids": list[str]}
    Returns:
        {"hit_rate": float, "mrr": float, "num_questions": int}
    """
    hit_rates, mrrs = [], [] 
    for qa in golden_qa:
        [query_embedding] = await pipeline._embedder.embed([qa["question"]])
        chunks = await pipeline._vector_store.search(query_embedding, k)
        retrieved_ids = [chunk.id for chunk in chunks]
        relevant_ids  = set(qa["relevant_chunk_ids"])
        hit_rates.append(hit_rate(retrieved_ids, relevant_ids))
        mrrs.append(mrr(retrieved_ids, relevant_ids))
    return {
            "hit_rate": sum(hit_rates) / len(hit_rates),
            "mrr": sum(mrrs) / len(mrrs),
            "num_questions": len(golden_qa),
        }
    
def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1.0 if any retrieved chunk is relevant else 0.0"""
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1/rank of the first relevant chunk or 0.0 in none found"""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0
