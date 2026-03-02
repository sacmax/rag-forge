from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_forge.config.settings import ChunkingConfig
from rag_forge.document.models import Document, Chunk

class RecursiveChunker:
    def __init__(self, config: ChunkingConfig):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    
    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []
        texts: list[str] = self._splitter.split_text(document.content)
        for i, text in enumerate(texts):
            chunk =  Chunk(
                content=text,
                document_id=document.id,
                chunk_index=i,
                source=document.source,
                metadata={**document.metadata}
            )
            chunks.append(chunk)
        return chunks
        