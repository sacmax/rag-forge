import asyncio

class LocalEmbedder:
    _DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def dimension(self) -> int:
        return self._DIMENSIONS.get(self._model_name, 384)
    
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._model.encode,texts)
        return result.tolist()