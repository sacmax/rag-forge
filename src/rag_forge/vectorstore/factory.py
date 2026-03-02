from rag_forge.config.settings import Settings
from rag_forge.core.interfaces import VectorStore
from rag_forge.vectorstore.chroma_store import ChromaStore

def create_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store.provider == "chroma":
        return ChromaStore(settings.vector_store)
    else:
        raise ValueError(f"Unknown vector store provider: {settings.vector_store.provider}")
    