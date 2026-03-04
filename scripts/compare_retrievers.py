#!/usr/bin/env python
"""Benchmark dense vs hybrid vs HyDE vs reranking on the golden QA dataset."""
import asyncio
import json
from pathlib import Path
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.evaluation.metrics import evaluate_retrieval

async def benchmark(label: str, overrides: dict, golden_qa: list[dict]) -> dict:
    settings = Settings()
    settings.retrieval.strategy = overrides["strategy"]
    settings.retrieval.reranker = overrides["reranker"]
    pipeline = RAGPipeline.from_settings(settings)
    results = await evaluate_retrieval(pipeline, golden_qa)
    return {"label": label, **results}

async def main() -> None:
    golden_qa = json.loads(Path("data/eval/golden_qa.json").read_text())

    configs = [
        ("dense (baseline)",    {"strategy": "dense",       "reranker": None}),
        ("dense + rerank",      {"strategy": "dense",       "reranker": "flashrank"}),
        ("hybrid",              {"strategy": "hybrid",      "reranker": None}),
        ("hybrid + rerank",     {"strategy": "hybrid",      "reranker": "flashrank"}),
        ("hyde",                {"strategy": "hyde",        "reranker": None}),
        ("hyde + rerank",       {"strategy": "hyde",        "reranker": "flashrank"}),
    ]
    results = []
    for label, overrides in configs:
        r = await benchmark(label, overrides, golden_qa)
        results.append(r)

    print(f"\n{'Strategy':<22} {'Hit Rate':>10} {'MRR':>8}")
    print("-" * 44)
    for r in results:
        print(f"{r['label']:<22} {r['hit_rate']:>10.3f} {r['mrr']:>8.3f}")

if __name__ == "__main__":
    asyncio.run(main())