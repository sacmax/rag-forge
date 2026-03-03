from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings

class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024

class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_forge"

class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_threshold: float = 0.85
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256

class Settings(BaseSettings):
    
    #API keys
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    # Nested config groups
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    chunking: ChunkingConfig = ChunkingConfig()

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__"
    }

