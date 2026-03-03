from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, TYPE_CHECKING
from collections.abc import AsyncIterator

if TYPE_CHECKING:
    from rag_forge.document.models import Document, Chunk


@runtime_checkable
class DocumentLoader(Protocol):
    async def load(self, path: str) -> list[Document]:
        """Load a file and return a list of Document objects"""
        ...

@runtime_checkable
class TextChunker(Protocol):
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into chunk objects"""
        ...

@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return the embedding vectors for a list of texts"""
        ...
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension"""
        ...
    
@runtime_checkable
class VectorStore(Protocol):
    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks and their embeddings"""
        ...

    async def search(self, query_embedding: list[float], k:int) -> list[Chunk]:
        """Return the k most similar chunks to the query embedding"""
        ...
    
    async def delete_collection(self) -> None:
        """Drop the entire collection - for testing and reingestion"""
        ...

@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return top_n chunks resorted by relevance to query"""
        ...

@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str:
        """Send a prompt and return the full response as a string"""
        ...

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens as they arrive"""
        ...
    
    @property
    def model(self) -> str:
        ...

@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> str | None:
        """Return cached value for key, or None if not found"""
        ...
    
    async def set(self, key: str, value: str) -> None:
        """Store a value under the given key"""
        ...

@runtime_checkable
class Tracer(Protocol):
    def trace(self, name: str, **kwargs: Any) -> Any:
        """Return a context manager for tracing a named span"""
        ...