from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.schemas.document import CollectionsResponse, DocumentUploadResponse
from services.rag_service import rag_service
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

router = APIRouter()

@router.get("/collections", response_model=CollectionsResponse)
async def get_collections():
    """Retrieve vector database collection names."""
    try:
        collections = settings.COLLECTION_NAMES
        return CollectionsResponse(collections=collections)
    except Exception as e:
        logger.error(f"Error fetching collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("Finance")
):
    """Upload and ingest a document into the specified ChromaDB collection."""
    try:
        logger.info(f"Uploading document '{file.filename}' to collection '{collection}'")
        file_bytes = await file.read()

        await rag_service.ingest_file(
            file_name=file.filename or "uploaded_document",
            file_bytes=file_bytes,
            file_type=file.content_type or "application/pdf",
            collection_name=collection
        )

        return DocumentUploadResponse(
            success=True,
            file_name=file.filename or "uploaded_document",
            collection=collection,
            message=f"Successfully ingested '{file.filename}' into collection '{collection}'."
        )
    except Exception as e:
        logger.error(f"Failed to ingest file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
