import time
from rag_forge.core.interfaces import TextChunker, Embedder, VectorStore, LLMClient
from rag_forge.config.settings import Settings
from rag_forge.generation.response import RAGResponse, Citation
from rag_forge.generation.prompts import build_rag_prompt
from rag_forge.document.loader import DocumentLoaderFactory
import structlog

logger = structlog.get_logger()

class RAGPipeline:
    def __init__(
            self,
            chunker: TextChunker,
            embedder: Embedder,
            vector_store: VectorStore,
            llm: LLMClient
    ):
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
    
    async def ingest(self, path: str) -> int:
        """Load, chunk, embed and store a document. Returns chunk count"""
        loader = DocumentLoaderFactory.get_loader(path)
        documents = await loader.load(path)
        chunks = [chunk for doc in documents for chunk in self._chunker.chunk(doc)]
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedder.embed(texts)
        await self._vector_store.add(chunks, embeddings)
        logger.info("ingested", path=path, chunks=len(chunks))
        return len(chunks)

    async def query(self, question: str, k: int=5) -> RAGResponse:
        """Retrieve relevant chunks and generate a grounded answer"""
        start = time.monotonic()
        [query_embedding] = await self._embedder.embed([question])
        chunks = await self._vector_store.search(query_embedding, k)
        prompt = build_rag_prompt(question, chunks)
        answer = await self._llm.complete(prompt)
        citations = [Citation(source=chunk.source, page=chunk.page, chunk_id=chunk.id) for chunk in chunks]
        latency_ms = (time.monotonic() - start) * 1000
        return RAGResponse(answer=answer, citations=citations, chunks_used=len(chunks), latency_ms=latency_ms, model=self._llm.model)
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Build a fully wired pipeline from config"""
        from rag_forge.embedding.factory import create_embedder
        from rag_forge.vectorstore.factory import create_vector_store
        from rag_forge.generation.llm_client import LiteLLMClient
        from rag_forge.document.chunker import RecursiveChunker

        embedder = create_embedder(settings)
        vector_store = create_vector_store(settings)
        llm = LiteLLMClient(settings.llm, settings.openai_api_key.get_secret_value())
        chunker = RecursiveChunker(settings.chunking)
        return cls(chunker=chunker, embedder=embedder, vector_store=vector_store, llm=llm)
        