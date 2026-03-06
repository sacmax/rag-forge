from typing import Any, Protocol, runtime_checkable
from rag_forge.document.models import Chunk
from rag_forge.generation.prompts import build_rag_prompt

@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    async def run(self, **kwargs) -> Any:
        ...

class SearchTool:
    name = "search"
    description = "Retrieve relevant chunks from the vector store for a query "

    def __init__(self, pipeline):
        self._pipeline = pipeline

    async def run(self, query: str, k: int = 5) -> list[Chunk]:
        # use advanced retriever if available else fallback to direct embed + search
        if self._pipeline._retriever:
            return await self._pipeline._retriever.search(query)
        [query_embedding] = await self._pipeline._embedder.embed([query])
        return await self._pipeline._vector_store.search(query_embedding, k)
    
    # direct answer tool - generates a final answer from chunks + question
class DirectAnswerTool:
    name = "direct_answer"
    description = "Generate an answer given retrieved context chunks"

    def __init__(self, pipeline):
        self._pipeline = pipeline
    
    async def run(self, chunks: list[Chunk], question: str) -> str:
        # build a rag prompt and call the llm
        prompt = build_rag_prompt(question, chunks)
        return await self._pipeline._llm.complete(prompt)
    