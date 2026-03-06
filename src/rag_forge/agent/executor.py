import json
from dataclasses import dataclass, field
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.core.interfaces import LLMClient
from rag_forge.document.models import Chunk
from rag_forge.agent.tools import SearchTool, DirectAnswerTool
from rag_forge.agent.router import QueryRouter, RouterStrategy
from rag_forge.agent.evaluator import AnswerEvaluator, Grade

REWRITE_PROMPT = """The following search query did not return sufficient information.

Original question: {original}
Query used: {current}
Retrieved chunk previews:
{chunk_previews}

Rewrite the search query to find more relevant information. Return ONLY the new query string, nothing else."""

@dataclass
class AgentStep:
    step: int
    query_used: str
    chunks_retrieved: int
    grade: str
    action: str

@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_hops: int = 0
    final_grade: str = ""

class AgentExecutor:
    def __init__(self, pipeline: RAGPipeline, llm: LLMClient, max_hops: int = 3):
        self._search = SearchTool(pipeline)
        self._answer = DirectAnswerTool(pipeline)
        self._router = QueryRouter(llm)
        self._evaluator = AnswerEvaluator(llm)
        self._llm = llm
        self._max_hops = max_hops

    async def run(self, question: str) -> AgentResult:
        steps = []
        strategy = await self._router.route(question)

        # direct answer, llm used its own knowledge, no retrieval
        if strategy == RouterStrategy.DIRECT:
            answer = await self._answer.run(chunks=[], question=question)
            return AgentResult(answer=answer, steps=[], total_hops=0, final_grade="DIRECT")
        # single hop or multi hop - retrieval loop required
        current_query = question
        chunks = []
        grade = Grade.AMBIGUOUS # default before first eval

        for hop in range(self._max_hops):
            # retrieve chunks using current query
            chunks = await self._search.run(query=current_query)

            # evaluate whether retrieved chunks answer the original question
            grade = await self._evaluator.evaluate(question, chunks)

            # decide next action
            if grade == Grade.CORRECT:
                action = "answer"
            elif hop == self._max_hops - 1:
                action = "give_up" # all hops exhausted, answer with best available
            else:
                action = "refine" # rewrite query and try again
            
            steps.append(AgentStep(
                step=hop + 1,
                query_used=current_query,
                chunks_retrieved=len(chunks),
                grade=grade.value,
                action=action
            ))

            if action in ("answer", "give_up"):
                break

            # rewrite query for next hop
            current_query = await self._rewrite_query(question, current_query, chunks)

        # generate final answer from the best chunks
        answer = await self._answer.run(chunks=chunks, question=question)
        return AgentResult(
            answer=answer,
            steps=steps,
            total_hops=len(steps),
            final_grade=grade.value,
        )
    async def _rewrite_query(self, original: str, current: str, chunks: list[Chunk]) -> str:
        chunk_previews = "\n".join(f"- {c.content[:100]}..." for c in chunks[:3])
        prompt = REWRITE_PROMPT.format(
            original=original,
            current=current,
            chunk_previews=chunk_previews,
        )
        # llm returns the new query string
        return await self._llm.complete(prompt)