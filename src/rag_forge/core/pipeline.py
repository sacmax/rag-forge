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
            llm: LLMClient,
            retriever=None,
            input_guard=None,
            relevance_gate=None,
            output_guard=None,
            semantic_cache=None,
            tracer=None
    ):
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._llm = llm
        self._retriever = retriever
        self._input_guard = input_guard
        self._relevance_gate = relevance_gate
        self._output_guard = output_guard
        self._semantic_cache = semantic_cache
        self._tracer = tracer
    
    async def ingest(self, path: str) -> int:
        """Load, chunk, embed and store a document. Returns chunk count"""
        chunks = []
        loader = DocumentLoaderFactory.get_loader(path)
        documents = await loader.load(path)
        # chunks = [chunk for doc in documents for chunk in self._chunker.chunk(doc)]
        for doc in documents:
            chunks.extend(await self._chunker.chunk(doc))
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embedder.embed(texts)
        await self._vector_store.add(chunks, embeddings)
        logger.info("ingested", path=path, chunks=len(chunks))
        return len(chunks)

    async def query(self, question: str, k: int=5) -> RAGResponse:
        """Retrieve relevant chunks and generate a grounded answer"""

        start = time.monotonic()

        # input validation
        if self._input_guard:
            is_valid, error_msg = self._input_guard.validate(question)
            if not is_valid:
                return RAGResponse(answer=error_msg, citations=[],chunks_used=0, latency_ms=0, model=self._llm.model)


        # check semantic cache
        if self._semantic_cache:
            cached = await self._semantic_cache.get(question)
            if cached:
                return cached
        
        # retrieve
        if self._retriever:
            chunks = await self._retriever.search(question)
        else:
            [query_embedding] = await self._embedder.embed([question])
            chunks = await self._vector_store.search(query_embedding, k)
        
        # relevance gate
        if self._relevance_gate:
            if not self._relevance_gate.check(chunks):
                return RAGResponse(
                    answer="I cannot find this information in the provided documents",
                    citations=[],chunks_used=0, latency_ms=(time.monotonic() - start) * 1000, model=self._llm.model)
            chunks = self._relevance_gate.filter(chunks)

        # generate
        prompt = build_rag_prompt(question, chunks)
        answer = await self._llm.complete(prompt)
        citations = [Citation(source=chunk.source, page=chunk.page, chunk_id=chunk.id) for chunk in chunks]
        latency_ms = (time.monotonic() - start) * 1000

        # scope check
        if self._output_guard:
            is_grounded, answer = await self._output_guard.check(answer, chunks, question)
        
        # build response 
        rag_response = RAGResponse(answer=answer, citations=citations, chunks_used=len(chunks), latency_ms=latency_ms, model=self._llm.model)

        # cache result
        if self._semantic_cache:
            await self._semantic_cache.set(question, rag_response)

        return rag_response
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Build a fully wired pipeline from config"""
        from rag_forge.embedding.factory import create_embedder
        from rag_forge.vectorstore.factory import create_vector_store
        from rag_forge.generation.llm_client import LiteLLMClient
        from rag_forge.document.chunker_factory import create_chunker
        from rag_forge.retrieval.retriever import AdvancedRetriever
        from rag_forge.retrieval.query_transform import HyDETransformer, MultiQueryTransformer
        from rag_forge.retrieval.reranker import create_reranker

        embedder = create_embedder(settings)
        vector_store = create_vector_store(settings)
        llm = LiteLLMClient(settings.llm, settings.openai_api_key.get_secret_value())
        chunker = create_chunker(settings, embedder)
    
        query_transformer = None
        if settings.retrieval.strategy == "hyde":
            query_transformer = HyDETransformer(llm)
        elif settings.retrieval.strategy == "multi_query":
            query_transformer = MultiQueryTransformer(llm, n=settings.retrieval.multi_query_n)
        
        reranker = create_reranker(settings.retrieval.reranker, cohere_api_key=settings.cohere_api_key.get_secret_value() if settings.cohere_api_key else None)

        retriever = AdvancedRetriever(vector_store, embedder, settings.retrieval, query_transformer=query_transformer, reranker=reranker)

        # wrap embedder in cache
        from rag_forge.cache.embedding_cache import EmbeddingCache
        if settings.cache.enabled:
            embedder = EmbeddingCache(embedder, max_size=10000)
        
        # build guardrails
        from rag_forge.guardrails.input_guard import InputGuard
        from rag_forge.guardrails.relevance import RelevanceGate
        from rag_forge.guardrails.output_guard import OutputGuard

        input_guard = InputGuard(max_length=settings.guardrails.max_query_length)
        relevance_gate = RelevanceGate(threshold=settings.guardrails.relevance_threshold)
        output_guard = OutputGuard(llm, enable_scope_check=settings.guardrails.enable_scope_check)

        # build semantic cache
        from rag_forge.cache.semantic_cache import SemanticCache

        semantic_cache = None
        if settings.cache.enabled:
            semantic_cache = SemanticCache(embedder, threshold=settings.cache.semantic_threshold, max_size=settings.cache.max_size, ttl_seconds=settings.cache.ttl_seconds)
        
        # build tracer
        from rag_forge.observability.tracer import PipelineTracer

        tracer = PipelineTracer(enable=settings.observability.enable_tracing, endpoint=settings.observability.phoenix_endpoint)

        return cls(chunker=chunker, embedder=embedder, vector_store=vector_store, llm=llm, retriever=retriever, input_guard=input_guard, relevance_gate=relevance_gate, output_guard=output_guard, semantic_cache=semantic_cache, tracer=tracer)

