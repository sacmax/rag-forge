import asyncio
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_forge.config.settings import ChunkingConfig
from rag_forge.core.interfaces import Embedder
from rag_forge.document.models import Document, Chunk


# helper function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    else:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Recursive Chunker
class RecursiveChunker:
    def __init__(self, config: ChunkingConfig):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    
    async def chunk(self, document: Document) -> list[Chunk]:
        chunks = []
        texts: list[str] = self._splitter.split_text(document.content)
        for i, text in enumerate(texts):
            chunk =  Chunk(
                content=text,
                document_id=document.id,
                chunk_index=i,
                source=document.source,
                page=document.metadata.get("page"),
                metadata={**document.metadata}
            )
            chunks.append(chunk)
        return chunks

# Semantic Chunker
class SemanticChunker:
    def __init__(self, embedder: Embedder, config: ChunkingConfig):
        self._embedder = embedder
        self._threshold = config.semantic_threshold
    
    async def chunk(self, document: Document) -> list[Chunk]:
        # split into sentences
        sentences = [s.strip() for s in document.content.split(". ") if s.strip()]
        if len(sentences) == 1:
            return [Chunk(content=document.content, document_id=document.id, chunk_index=0, source=document.source,page=document.metadata.get("page"), metadata={**document.metadata})]
        
        # embed all sentences in one call
        embeddings = await self._embedder.embed(sentences)
        emb_array = np.array(embeddings)

        # find breakpoints - where to split
        breakpoints = []
        for i in range(len(emb_array) - 1):
            sim = cosine_similarity(emb_array[i], emb_array[i + 1])
            if sim < self._threshold:
                breakpoints.append(i)
        
        # group sentences between breakpoints into chunks
        groups = []
        start = 0
        for bp in breakpoints:
            groups.append(sentences[start : bp + 1])
            start = bp + 1
        groups.append(sentences[start:]) # last gp
        texts = [". ".join(g) + "." for g in groups]

        # create chunk objects
        chunks =[]
        for i, text in enumerate(texts):
            chunks.append(Chunk(content=text, document_id=document.id, chunk_index=i, source=document.source,page=document.metadata.get("page"), metadata={**document.metadata}))
        return chunks

# Hybrid Chunker
class HybridChunker:
    def __init__(self, config: ChunkingConfig):
        # parent splitter(large chunks, no overlap)
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # child splitter(small chunks with overlap)
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._parents: dict[str, Chunk] = {}

    async def chunk(self, document: Document) -> list[Chunk]:
        # split into parent texts
        parent_texts = self._parent_splitter.split_text(document.content)

        # for each parent text, create a parent chunk
        child_chunks = []
        global_index = 0
        for i, parent_text in enumerate(parent_texts):
            parent = Chunk(content=parent_text, document_id=document.id, chunk_index=i, source=document.source, parent_id=None,page=document.metadata.get("page"), metadata={**document.metadata})
            self._parents[parent.id] = parent

        # split each parent into child chunks
            child_texts = self._child_splitter.split_text(parent.content)
            for j, child_text in enumerate(child_texts):
                child_chunks.append(Chunk(content=child_text, document_id=document.id, chunk_index=global_index, source=document.source, parent_id=parent.id,page=document.metadata.get("page"), metadata={**document.metadata}))
                global_index += 1
        return child_chunks
    
    def get_parent(self, parent_id: str) -> Chunk | None:
        # instead of sending child to LLM, this fn is used to send parent
        return self._parents.get(parent_id)




        
        
