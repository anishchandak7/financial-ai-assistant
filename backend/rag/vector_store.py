import chromadb
from typing import Dict
from chromadb.api.models.Collection import Collection
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

class ChromDBVectorStore:
    
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    
    def check_status(self):
        try:
            return self.client.heartbeat()
        except Exception as e:
            logger.error(msg=f"Chromadb vector store at path {settings.CHROMA_DB_PATH} is not responding due to some issue.", exc_info=True, stack_info=True)
            raise e
    
    def setup_collections(self) -> Dict[str, Collection]:
        try:
            collections = settings.COLLECTION_NAMES
            collection_mapping = {}
            for collection in collections:
                created_collection = self.client.get_or_create_collection(
                    name=collection,
                    configuration={
                        "hnsw":{
                            "space": "cosine",
                            "ef_construction": 200
                        }
                    }
                )
                logger.info(msg=f"{collection} created successfully.")
                collection_mapping[collection] = created_collection
            return collection_mapping
        except Exception as e:
            logger.error(msg="Failed to create collection in the index.", exc_info=True, stack_info=True)
            raise e

vector_store = ChromDBVectorStore()