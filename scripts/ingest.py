#!/usr/bin/env python
"""CLI: ingest a document into the vector store"""
import asyncio, sys
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline

async def main(path: str) -> None:
    settings = Settings()
    pipeline = RAGPipeline.from_settings(settings)
    count = await pipeline.ingest(path)
    print(f"Ingested {count} chunks from {path}")

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))