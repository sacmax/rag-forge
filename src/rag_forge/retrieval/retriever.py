import numpy as np
from rag_forge.core.interfaces import Embedder, VectorStore, Reranker
from rag_forge.document.models import Chunk
from rag_forge.config.settings import RetrievalConfig
from rag_forge.retrieval.query_transform import HyDETransformer, MultiQueryTransformer
from rag_forge.retrieval.hybrid import HybridRetriever

def mmr(
        query_embedding: list[float],
        chunks: list[Chunk],
        chunk_embeddings: list[list[float]],
        k: int,
        lambda_mult: float = 0.5,
) -> list[Chunk]:
    """
    Maximal Marginal Relevance - selects diverse and relevant chunks
    Args:
        query_embedding: the query vector
        chunks: candidate chunks(pre-retrieved)
        chunk_embeddings: embedding of each candidate chunk
        k: number of chunks to select
        lambda_mult: relevance vs diversity tradeoff(1 = relevance only)
    """
    query_vec = np.array(query_embedding)
    emb_matrix = np.array(chunk_embeddings)
    selected_indices = []
    remaining = list(range(len(chunks)))

    while len(selected_indices) < k and remaining:
        if not selected_indices:
            # pick most relevant to query
            relevance = emb_matrix[remaining] @ query_vec
            best = remaining[np.argmax(relevance)]
        else:
            # balance relevance and diversity
            relevance = emb_matrix[remaining] @ query_vec
            selected_embs = emb_matrix[selected_indices]

            # max similarity to any already selected chunk
            redundancy = (emb_matrix[remaining] @ selected_embs.T).max(axis=1)
            scores = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            best = remaining[np.argmax(scores)]
        selected_indices.append(best)
        remaining.remove(best)
    return [chunks[i] for i in selected_indices]

class AdvancedRetriever:
    def __init__(
            self,
            vector_store: VectorStore,
            embedder: Embedder,
            config: RetrievalConfig,              # Retrieval Config
            query_transformer=None, # HyDeTransformer, MultiqueryTransformer , None
            hybrid_retriever=None, # HybridRetriever, None
            reranker: Reranker | None = None,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._config = config
        self._query_transformer = query_transformer
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker

    async def search(self, query: str) -> list[Chunk]:
        # query transformation
        if isinstance(self._query_transformer, HyDETransformer):
            hypothetical = await self._query_transformer.transform(query)
            # dense retrieval
            [query_emb] = await self._embedder.embed([hypothetical])
            chunks = await self._vector_store.search(query_emb, k=self._config.k)
        elif isinstance(self._query_transformer, MultiQueryTransformer):
            queries = await self._query_transformer.transform(query)
            seen_ids = set()
            chunks = []
            for q in queries:
                [q_emb] = await self._embedder.embed([q])
                results = await self._vector_store.search(q_emb, k=self._config.k)
                for chunk in results:
                    if chunk.id not in seen_ids:
                        seen_ids.add(chunk.id)
                        chunks.append(chunk)
        else:
            [query_emb] = await self._embedder.embed([query])
            chunks = await self._vector_store.search(query_emb, k=self._config.k)
    
        # hybrid fusion
        if isinstance(self._hybrid_retriever, HybridRetriever):
            dense_ids = [c.id for c in chunks]
            chunks = self._hybrid_retriever.fuse(query, dense_ids)[:self._config.k]
        
        # reranking
        if self._reranker:
            chunks = await self._reranker.rerank(query, chunks, self._config.top_n)

        # mmr diversification
        if self._config.use_mmr and len(chunks) > self._config.top_n:
            if not isinstance(self._query_transformer, MultiQueryTransformer):
                # query_emb already exists from HyDE or default path
                pass
            else:
                [query_emb] = await self._embedder.embed([query])
            embeddings = await self._embedder.embed([c.content for c in chunks])
            chunks = mmr(query_emb, chunks, embeddings, self._config.top_n, self._config.mmr_lambda)

        return chunks