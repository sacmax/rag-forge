from rank_bm25 import BM25Okapi
from rag_forge.document.models import Chunk

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists of chunk IDs using RRF
    Args:
        ranked_lists: each inner list is chunk IDs in rank order(best first)
        k: RRF constant - higher k reduces the impact of top ranks
    Returns:
        Merged list of chunk IDs sorted by fused score(best first)
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)

class HybridRetriever:
    def __init__(self, chunks: list[Chunk], rrf_k: int = 60):
        tokenized = [chunk.content.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunks = chunks
        self._chunk_index = {chunk.id: chunk for chunk in chunks}
        self._rrf_k = rrf_k
    
    def bm25_rank(self, query: str) -> list[str]:
        """Return chunk IDs ranked by BM25 score"""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i].id for i in ranked_indices]
    
    def fuse(self, query: str, dense_ranked_ids: list[str]) -> list[Chunk]:
        """Fuse BM25 and dense ranked lists using RRF, return chunk objects"""
        bm25_ids = self.bm25_rank(query)
        fused_ids = reciprocal_rank_fusion([dense_ranked_ids, bm25_ids], k=self._rrf_k)
        return [self._chunk_index[cid] for cid in fused_ids if cid in self._chunk_index]
    