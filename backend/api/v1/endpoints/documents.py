from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from api.schemas.document import (
    CollectionsResponse,
    CollectionStat,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentDeleteResponse,
    CollectionClearResponse
)
from services.rag_service import rag_service
from rag.vector_store import vector_store, FinancialCollection
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()


@router.get("/collections", response_model=CollectionsResponse)
async def get_collections():
    """Retrieve vector database collection names and health statistics."""
    try:
        configured = settings.COLLECTION_NAMES
        stats_raw = vector_store.get_collection_stats()
        stats = [
            CollectionStat(
                collection=s["collection"],
                chunk_count=s["chunk_count"],
                status=s["status"]
            )
            for s in stats_raw
        ]
        return CollectionsResponse(collections=configured, stats=stats)
    except Exception as e:
        logger.error(f"Error fetching collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection}/stats")
async def get_single_collection_stats(collection: str):
    """Retrieve chunk count and health for a specific collection."""
    try:
        canonical = FinancialCollection.from_str(collection).value
        col = vector_store.get_collection(canonical)
        return {
            "collection": canonical,
            "chunk_count": col.count(),
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Error fetching stats for '{collection}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("Finance")
):
    """Upload and ingest a financial document (.pdf, .txt, .csv) with chunking and metadata extraction."""
    try:
        file_name = file.filename or "uploaded_document"
        logger.info(f"Uploading document '{file_name}' to collection '{collection}'")
        file_bytes = await file.read()

        result = await rag_service.ingest_file(
            file_name=file_name,
            file_bytes=file_bytes,
            file_type=file.content_type or "application/pdf",
            collection_name=collection
        )

        return DocumentUploadResponse(
            success=True,
            file_id=result["file_id"],
            file_name=result["file_name"],
            collection=result["collection"],
            chunk_count=result["chunk_count"],
            message=f"Successfully ingested '{file_name}' ({result['chunk_count']} chunks) into collection '{result['collection']}'."
        )
    except Exception as e:
        logger.error(f"Failed to ingest file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(collection: Optional[str] = Query(None, description="Filter by collection name")):
    """List indexed documents across ChromaDB collections with their metadata and chunk counts."""
    try:
        docs_raw = await rag_service.list_documents(collection_name=collection)
        docs = [DocumentMetadata(**d) for d in docs_raw]
        return DocumentListResponse(documents=docs, total_documents=len(docs))
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{file_id}", response_model=DocumentDeleteResponse)
async def delete_document(file_id: str):
    """Delete a document and all its chunks from the vector store."""
    try:
        deleted_count = await rag_service.delete_document(file_id)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document with file_id '{file_id}' not found.")
        return DocumentDeleteResponse(
            success=True,
            file_id=file_id,
            deleted_chunks=deleted_count,
            message=f"Successfully deleted document '{file_id}' and {deleted_count} chunks from ChromaDB."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document '{file_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection}/clear", response_model=CollectionClearResponse)
async def clear_collection(collection: str):
    """Purge all document chunks from a specific collection."""
    try:
        canonical = FinancialCollection.from_str(collection).value
        vector_store.clear_collection(canonical)
        return CollectionClearResponse(
            success=True,
            collection=canonical,
            message=f"Collection '{canonical}' cleared successfully."
        )
    except Exception as e:
        logger.error(f"Error clearing collection '{collection}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
