from rag_forge.core.interfaces import LLMClient
from rag_forge.document.models import Chunk

SCOPE_CHECK_PROMPT = """\
You are a fact-checking assistant. Given a set of context passages and an answer, determine if EVERY factual claim in the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Respond with ONLY one of:
- "GROUNDED" - if every claim is supported by the context
- "NOT_GROUNDED: <brief explanation>" - if any claim goes beyond the context
"""

class OutputGuard:
    def __init__(self, llm: LLMClient, enable_scope_check: bool = True):
        self._llm = llm
        self._enable_scope_check = enable_scope_check
    
    async def check(self, answer: str, chunks: list[Chunk], query: str) -> tuple[bool, str]:
        """
        Return (is_grounded, response)
        if grounded, response = original answer
        if not grounder, response = refusal message
        """
        if not self._enable_scope_check:
            return True, answer
        
        context = "\n\n".join(c.content for c in chunks)
        prompt = SCOPE_CHECK_PROMPT.format(context=context, answer=answer)
        result = await self._llm.complete(prompt)

        if result.strip().startswith("GROUNDED"):
            return True, answer
        else:
            return False, "I cannot verify this answer against the provided documents. " + answer