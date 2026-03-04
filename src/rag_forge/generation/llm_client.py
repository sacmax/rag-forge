import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from collections.abc import AsyncIterator
from rag_forge.config.settings import LLMConfig

class CircuitBreaker:
    """Simple circuit breaker - opens after max_failures consecutive errors"""
    
    def __init__(self, max_failures: int = 3):
        self._max_failures = max_failures
        self._failure_count = 0
        self._is_open = False

    def record_success(self) -> None:
        self._failure_count = 0
        self._is_open = False
    
    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._max_failures:
            self._is_open = True 
    
    @property
    def is_open(self) -> bool:
        return self._is_open

class LiteLLMClient:
    def __init__(self, config: LLMConfig, api_key: str, fallback_model: str | None = None, fallback_api_key: str | None = None):
        self._model = f"{config.provider}/{config.model}"
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._fallback_model = fallback_model
        self._fallback_api_key = fallback_api_key
        self._circuit_breaker = CircuitBreaker()

        if config.provider == "openai":
            litellm.openai_key = api_key
        elif config.provider == "anthropic":
            litellm.anthropic_key = api_key
        else:
            raise ValueError(f"unknown LLM provider: {config.provider}")

    @property
    def model(self) -> str:
        return self._model
    
    async def _call(self, model: str, prompt: str) -> str:
        """Make a single llm call"""
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete(self, prompt: str) -> str:
        """"Complete with retry and circuit breaker fallback"""
        if self._circuit_breaker.is_open and self._fallback_model:
            return await self._call(self._fallback_model, prompt)
        
        try:
            result = await self._call(self._model, prompt)
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            self._circuit_breaker.record_failure()
            if self._circuit_breaker.is_open and self._fallback_model:
                return await self._call(self._fallback_model, prompt)
            raise
    
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


    

