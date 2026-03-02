from openai import AsyncOpenAI
from rag_forge.config.settings import EmbeddingConfig

class OpenAIEmbedder:
    def __init__(self, config: EmbeddingConfig, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = config.model
        self._batch_size = config.batch_size
        self._dimension = config.dimension

    @property
    def dimension(self) -> int:
        return self._dimension
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = await self._client.embeddings.create(input=batch, model=self._model)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
