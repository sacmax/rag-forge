#!/usr/bin/env python
"""CLI: run retrieval evaluation against golden QA dataset"""
import asyncio, json
from pathlib import Path
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.evaluation.metrics import evaluate_retrieval

async def main() -> None:
    settings = Settings()
    pipeline = RAGPipeline.from_settings(settings)
    golden_qa = json.loads(Path("data/eval/golden_qa.json").read_text())
    results = await evaluate_retrieval(pipeline, golden_qa)
    print(f"Hit Rate: {results['hit_rate']:.3f}")
    print(f"MRR:      {results['mrr']:.3f}")
    print(f"Questions: {results['num_questions']}")

if __name__ == "__main__":
    asyncio.run(main())
