import time
import numpy as np
from rag_forge.core.interfaces import Embedder
from rag_forge.generation.response import RAGResponse

class SemanticCache:
    def __init__(self, embedder: Embedder, threshold: float = 0.95, max_size: int = 1000, ttl_seconds: int = 3600):
        self._embedder = embedder
        self._threshold = threshold
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._entries: list[dict] = []

    async def get(self, query: str) -> RAGResponse | None:
        """Return cached response if semantically similar query exists else None"""
        [query_emb] = await self._embedder.embed([query])
        query_vec = np.array(query_emb)
        now = time.time()

        for entry in self._entries:
            # skip expired entries
            if now - entry["timestamp"] > self._ttl:
                continue
            cached_vec = np.array(entry["embedding"])
            similarity = np.dot(query_vec, cached_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec))
            if similarity >= self._threshold:
                return entry["response"]
        return None
    
    async def set(self, query: str, response: RAGResponse) -> None:
        """Cache a query response pair"""
        [query_emb] = await self._embedder.embed([query])
        self._entries.append({
            "embedding": query_emb,
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
        # evict oldest if over max size
        if len(self._entries) > self._max_size:
            self._entries = self._entries[-self._max_size:]
        
    def clear(self) -> None:
        """clear all cached entries"""
        self._entries.clear()
