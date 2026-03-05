from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag_forge.config.settings import VectorStoreConfig
from rag_forge.document.models import Chunk
import uuid

class QdrantStore:
    def __init__(self, config: VectorStoreConfig):
        self._client = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self._collection = config.collection_name
        self._dimension = config.dimension

    async def _ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            await self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE)
            )

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        await self._ensure_collection()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"chunk_id": chunk.id, "content": chunk.content, "source": chunk.source, "page": chunk.page or 0, "document_id": chunk.document_id, "chunk_index": chunk.chunk_index}
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
    
    async def search(self, query_embeddings: list[float], k: int) -> list[Chunk]:
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_embeddings,
            limit=k,
            with_payload=True
        )
        return [
            Chunk(
                id=r.payload["chunk_id"],
                content=r.payload["content"],
                document_id=r.payload["document_id"],
                chunk_index=r.payload["chunk_index"],
                source=r.payload["source"],
                page=r.payload["page"] or None,
                metadata={"score": r.score}
            )
            for r in results
        ]