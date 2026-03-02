import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from collections.abc import AsyncIterator
from rag_forge.config.settings import LLMConfig

class LiteLLMClient:
    def __init__(self, config: LLMConfig, api_key: str):
        self._model = f"{config.provider}/{config.model}"
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        if config.provider == "openai":
            litellm.openai_key = api_key
        elif config.provider == "anthropic":
            litellm.anthropic_key = api_key
        else:
            raise ValueError(f"unknown LLM provider: {config.provider}")

    @property
    def model(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete(self, prompt: str) -> str:
        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content
    
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

