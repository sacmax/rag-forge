from rag_forge.core.interfaces import LLMClient

HYDE_PROMPT = """\
Write a short passage (3-5 sentences) that directly answers the following question. Write as if you are the document that contains this information.
Question: {query}
Passage:"""

MULTI_QUERY_PROMPT = """\
Generate {n} different phrasings of the following question. Return one question per line, no numbering or bullet points.
Question: {query}
Paraphrases:"""

class HyDETransformer:
    def __init__(self, llm: LLMClient):
        self._llm = llm
    
    async def transform(self, query: str) -> str:
        """Return a hypothetical document that would answer the query"""
        prompt = HYDE_PROMPT.format(query=query)
        return await self._llm.complete(prompt)
    
class MultiQueryTransformer:
    def __init__(self, llm: LLMClient, n: int = 3):
        self._llm = llm
        self._n = n

    async def transform(self, query: str) -> list[str]:
        """Return n paraphrases of the query, including the original"""
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=self._n)
        response = await self._llm.complete(prompt)
        paraphrases = [line.strip() for line in response.splitlines() if line.strip()]
        return [query] + paraphrases[:self._n]