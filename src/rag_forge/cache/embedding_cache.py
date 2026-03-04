import hashlib
from rag_forge.core.interfaces import Embedder

class EmbeddingCache:
    """Wraps an embedder, caches results by text content hash"""

    def __init__(self, embedder: Embedder, max_size: int = 10000):
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_size

    @property
    def dimension(self) -> int:
        return self._embedder.dimension
    
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            h = self._hash(text)
            if h in self._cache:
                results[i] = self._cache[h]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            new_embeddings = await self._embedder.embed(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                h = self._hash(texts[idx])
                self._cache[h] = emb
                results[idx] = emb
            if len(self._cache) > self._max_size:
                # evict the oldest half
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]
        return results