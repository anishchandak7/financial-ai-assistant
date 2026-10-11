from typing import List
from pydantic import BaseModel, Field

class CollectionsResponse(BaseModel):
    collections: List[str]

class DocumentUploadResponse(BaseModel):
    success: bool
    file_name: str
    collection: str
    message: str
