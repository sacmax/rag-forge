from rag_forge.document.models import Chunk

class RelevanceGate:
    def __init__(self, threshold: float = 0.3):
        self._threshold = threshold

    def check(self, chunks: list[Chunk]) -> bool:
        """Return true if at least one chunk is above the relevance threshold"""
        return any(chunk.metadata.get("score", 0) >= self._threshold for chunk in chunks)
    
    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return only chunks with score >= threshold"""
        return [c for c in chunks if c.metadata.get("score", 0) >= self._threshold]
