#!/usr/bin/env python
"""CLI: query the RAG system"""
import asyncio, sys
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline

async def main(question: str) -> None:
    settings = Settings()
    pipeline = RAGPipeline.from_settings(settings)
    response = await pipeline.query(question)
    print(response.answer)
    print("\n Sources:")
    for citation in response.citations:
        print(f" - {citation.source}, page {citation.page}")

if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))