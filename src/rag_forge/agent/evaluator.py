import json
from enum import Enum
from rag_forge.core.interfaces import LLMClient
from rag_forge.document.models import Chunk

class Grade(Enum):
    CORRECT = "CORRECT" # chunks contain clear direct answer
    INCORRECT = "INCORRECT" # chunks are off topic or contradictory
    AMBIGUOUS = "AMBIGUOUS" # chunks are partially relevant, more retrieval may help

EVAL_PROMPT = """You are evaluating whether retrieved document chunks answer a user question.

Question: {question}

Retrieved chunks:
{chunks_text}

Grade the retrieval result:
- CORRECT: the chunks contain a clear, direct answer to the question
- INCORRECT: the chunks are off-topic or don't contain the answer at all
- AMBIGUOUS: the chunks are partially relevant; more information may help

Respond with JSON only, no prose: {{"grade": "CORRECT" | "INCORRECT" | "AMBIGUOUS", "reason": "one sentence"}}"""

class AnswerEvaluator:
    def __init__(self, llm: LLMClient):
        self._llm = llm
    
    async def evaluate(self, question: str, chunks: list[Chunk]) -> Grade:
        chunks_text = "\n---\n".join(c.content for c in chunks)
        prompt = EVAL_PROMPT.format(question=question, chunks_text=chunks_text)
        response = await self._llm.complete(prompt)
        try:
            data = json.loads(response)
            return Grade(data["grade"])
        except Exception:
            return Grade.AMBIGUOUS
        

