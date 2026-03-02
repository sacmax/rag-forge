from rag_forge.config.settings import Settings
from rag_forge.core.interfaces import Embedder
from rag_forge.embedding.openai_embedder import OpenAIEmbedder
from rag_forge.embedding.local_embedder import LocalEmbedder

def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding.provider == "openai":
        return OpenAIEmbedder(settings.embedding, settings.openai_api_key.get_secret_value())
    elif settings.embedding.provider == "local":
        return LocalEmbedder()
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding.provider}")
    