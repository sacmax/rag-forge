from __future__ import annotations
import uuid
from typing import Any
from pydantic import BaseModel,Field

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: str
    file_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    document_id: str
    chunk_index: int
    source: str
    page: int | None = None
    parent_id: str | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)