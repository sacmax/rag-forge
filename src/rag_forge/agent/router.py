import json
from enum import Enum
from rag_forge.core.interfaces import LLMClient

# three routing strategies
class RouterStrategy(Enum):
    DIRECT = "DIRECT" # llm answers from its own knowledge, no retrieval required
    SINGLE_HOP = "SINGLE_HOP" # one retrieval ste, standard rag
    MULTI_HOP = "MULTI_HOP" # cross doc comparison, needs multiple retrievals

ROUTER_PROMPT = """You are a query router for a RAG system. Given the user query, classify it:
- DIRECT: simple factual question the LLM can answer without documents (e.g. "What is Python?")
- SINGLE_HOP: requires retrieving from one document or one concept (e.g. "What are the payment terms?")
- MULTI_HOP: requires comparing, aggregating, or reasoning across multiple retrievals (e.g. "Compare clause 3 in contract A vs contract B")

Respond with JSON only, no prose: {{"strategy": "DIRECT" | "SINGLE_HOP" | "MULTI_HOP", "reason": "one sentence"}}

Query: {query}"""

class QueryRouter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def route(self, query: str) -> RouterStrategy:
        prompt = ROUTER_PROMPT.format(query=query)
        response = await self._llm.complete(prompt)
        try:
            data = json.loads(response)
            return RouterStrategy(data["strategy"])
        except Exception:
            return RouterStrategy.SINGLE_HOP


