#!/usr/bin/env python
"""Benchmark recursive vs semantic vs hybrid chunking on the golden QA dataset"""
import asyncio
import json
from pathlib import Path
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.evaluation.metrics import evaluate_retrieval


async def benchmark(strategy: str, doc_path: str, golden_qa: list[dict]) -> dict:
    # load settings, override chunking strategy and collection name
    settings = Settings()
    settings.chunking.strategy = strategy
    settings.vector_store.collection_name = f"benchmark_{strategy}"

    # build pipeline with from_settings()
    pipeline = RAGPipeline.from_settings(settings)

    # ingest document
    chunk_count = await pipeline.ingest(doc_path)

    # evaluate
    results = await evaluate_retrieval(pipeline, golden_qa)

    # delete test collection from chroma db
    await pipeline._vector_store.delete_collection()

    # retun results
    return {"strategy": strategy, "chunks": chunk_count, **results}

async def main() -> None:
    doc_path = "data/raw/sample.pdf"
    golden_qa = json.loads(Path("data/eval/golden_qa.json").read_text())
    strategies = ["recursive", "semantic", "hybrid"]

    results = []
    # run the benchmark sequentially 
    for strategy in strategies:
        r = await benchmark(strategy, doc_path, golden_qa)
        results.append(r)
    
    # print comparison table
    print(f"\n{'Strategy':<12} {'Chunks':>6} {'Hit Rate':>10} {'MRR':>8}")
    print("-" * 40)
    for r in results:
        print(f"{r['strategy']:<12} {r['chunks']:>6} {r['hit_rate']:>10.3f} {r['mrr']:>8.3f}")
    
if __name__ == "__main__":
    asyncio.run(main())

