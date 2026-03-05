from rag_forge.config.settings import Settings
from rag_forge.core.interfaces import VectorStore



def create_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store.provider == "chroma":
        from rag_forge.vectorstore.chroma_store import ChromaStore
        return ChromaStore(settings.vector_store)
    elif settings.vector_store.provider == "qdrant":
        from rag_forge.vectorstore.qdrant_store import QdrantStore
        return QdrantStore(settings.vector_store)
    else:
        raise ValueError(f"Unknown vector store provider: {settings.vector_store.provider}")
    