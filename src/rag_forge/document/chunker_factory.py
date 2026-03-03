from rag_forge.config.settings import Settings
from rag_forge.core.interfaces import Embedder, TextChunker
from rag_forge.document.chunker import RecursiveChunker, SemanticChunker, HybridChunker

def create_chunker(settings: Settings, embedder: Embedder | None = None) -> TextChunker:
    strategy = settings.chunking.strategy
    if strategy == "recursive":
        return RecursiveChunker(settings.chunking)
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("SemanticChunker requires an embedder - provide embbder to create_chunker()")
        return SemanticChunker(embedder, settings.chunking)
    if strategy == "hybrid":
        return HybridChunker(settings.chunking)
    raise ValueError(f"Unknown chunking strategy: {strategy}")