from rag_forge.document.models import Chunk
import asyncio

class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        import cohere
        self._client = cohere.AsyncClient(api_key)
        self._model = model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        response = await self._client.rerank(
            query=query,
            documents=[chunk.content for chunk in chunks],
            top_n=top_n,
            model=self._model,    
            )
        return [chunks[r.index] for r in response.results]
    
class FlashRankReranker:
    def __init__(self, model: str = "ms-marco-MiniLM-L-12-v2"):
        from flashrank import Ranker, RerankRequest
        self._ranker = Ranker(model_name=model)

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": chunk.content} for i, chunk in enumerate(chunks)]
        request = RerankRequest(query=query, passages=passages)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._ranker.rerank, request)
        return [chunks[r["id"]] for r in results[:top_n]]
    
class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model
        self._model = None
    
    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        pairs = [[query, chunk.content] for chunk in chunks]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        ranked = sorted(zip(scores, chunks), reverse=True)
        return [chunk for _, chunk in ranked[:top_n]]

def create_reranker(strategy: str, cohere_api_key: str | None = None):
    """Factory - return the right reranker or None"""
    if strategy == "cohere":
        if cohere_api_key is None:
            raise ValueError("Cohere reranker requires cohere_api_key")
        return CohereReranker(cohere_api_key)
    if strategy == "flashrank":
        return FlashRankReranker()
    if strategy == "cross_encoder":
        return CrossEncoderReranker()
    if strategy is None or strategy == "none":
        return None
    raise ValueError(f"Unknown reranker: {strategy}")

