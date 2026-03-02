import chromadb
from rag_forge.config.settings import VectorStoreConfig
from rag_forge.document.models import Chunk
import asyncio

class ChromaStore:
    def __init__(self, config: VectorStoreConfig):
        self._client = chromadb.PersistentClient(path=config.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [{"source": chunk.source, "page":chunk.page or 0, "document_id": chunk.document_id, "chunk_index": chunk.chunk_index} for chunk in chunks]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings))
        
    async def search(self, query_embedding: list[float], k: int) -> list[Chunk]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, lambda: self._collection.query(query_embeddings=[query_embedding], n_results=k))
        chunk_list = []
        for id_, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = Chunk(
                id=id_,
                content=text,
                document_id=meta["document_id"],
                chunk_index=meta["chunk_index"],
                source=meta["source"],
                page=meta["page"] or None,
                metadata={**meta, "score": 1 - dist},
            )
            chunk_list.append(chunk)
        chunk_list.sort(key=lambda chunk: chunk.metadata["score"], reverse=True)
        return chunk_list
    
    async def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)
            