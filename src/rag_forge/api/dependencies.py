from functools import lru_cache
from rag_forge.core.pipeline import RAGPipeline
from rag_forge.config.settings import Settings

@lru_cache
def get_settings() -> Settings:
    # lru cache - load settings once
    return Settings()

def get_pipeline() -> RAGPipeline:
    # called once at startup, store in app.state
    # routes access it - pipeline = Depends(get_pipeline)
    settings = get_settings()
    return RAGPipeline.from_settings(settings)
