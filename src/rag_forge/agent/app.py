import sys
import asyncio
from rag_forge.config.settings import Settings
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.generation.llm_client import LiteLLMClient
from rag_forge.agent.executor import AgentExecutor

async def main(question: str) -> None:
    # laod settings from .env
    settings = Settings()

    # build the full pipeline(chunker, embedder, vector store, retriever etc)
    pipeline = RAGPipeline.from_settings(settings)

    # build a separate llm client for agent's router/evaluator/reqriter
    llm = LiteLLMClient(settings.llm, settings.openai_api_key.get_secret_value())

    # build the executor
    executor = AgentExecutor(pipeline=pipeline, llm=llm, max_hops=3)

    # run the agent
    result = await executor.run(question)

    # print step trace
    for step in result.steps:
        print(f"Hop {step.step}: query=\"{step.query_used}\" → {step.chunks_retrieved} chunks → {step.grade} → {step.action}")
    
    print(f"\nAnswer ({result.total_hops} hop(s), final grade: {result.final_grade}):")
    print(result.answer)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m rag_forge.agent.app \"your question here\"")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))