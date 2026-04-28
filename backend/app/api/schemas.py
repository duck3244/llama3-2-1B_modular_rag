from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ready: bool
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class DocumentRef(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[DocumentRef] = Field(default_factory=list)
    cached: bool = False
    elapsed_ms: int = 0


class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunks_indexed: bool = True
