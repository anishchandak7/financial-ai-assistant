from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CollectionStat(BaseModel):
    collection: str
    chunk_count: int
    status: str


class CollectionsResponse(BaseModel):
    collections: List[str]
    stats: Optional[List[CollectionStat]] = None


class DocumentMetadata(BaseModel):
    file_id: str
    file_name: str
    file_type: str
    file_hash: str
    collection: str
    chunk_count: int
    total_pages: int = 1
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[str] = None
    upload_timestamp: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total_documents: int


class DocumentUploadResponse(BaseModel):
    success: bool
    file_id: Optional[str] = None
    file_name: str
    collection: str
    chunk_count: Optional[int] = 0
    message: str


class DocumentDeleteResponse(BaseModel):
    success: bool
    file_id: str
    deleted_chunks: int
    message: str


class CollectionClearResponse(BaseModel):
    success: bool
    collection: str
    message: str
