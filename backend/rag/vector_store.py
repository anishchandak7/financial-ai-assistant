"""Thread-Safe ChromaDB Vector Store Client and Collection Manager.

Provides type-safe collection management, concurrency control via threading locks,
HNSW index parameter verification, collection health statistics, and document deletion.
"""

from enum import Enum
import threading
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)


class FinancialCollection(str, Enum):
    """Canonical vector collection names in the Financial AI Assistant."""
    FINANCE = "Finance"
    MARKET_ANALYSIS = "MarketAnalysis"
    PORTFOLIO = "Portfolio"
    GOALS = "Goals"
    NEWS = "News"
    TAX = "Tax"

    @classmethod
    def from_str(cls, value: Any) -> "FinancialCollection":
        """Resolve arbitrary string or enum input into a canonical collection enum."""
        if isinstance(value, cls):
            return value
        raw_str = value.value if hasattr(value, "value") else str(value or "")
        clean = raw_str.strip().lower()
        if "financialcollection." in clean:
            clean = clean.split("financialcollection.")[-1]
        mapping = {
            "finance": cls.FINANCE,
            "marketanalysis": cls.MARKET_ANALYSIS,
            "market_analysis": cls.MARKET_ANALYSIS,
            "market": cls.MARKET_ANALYSIS,
            "portfolio": cls.PORTFOLIO,
            "goals": cls.GOALS,
            "goal": cls.GOALS,
            "news": cls.NEWS,
            "tax": cls.TAX,
        }
        if clean in mapping:
            return mapping[clean]
        for member in cls:
            if member.value.lower() == clean:
                return member
        logger.warning(f"Unrecognized collection name '{value}', falling back to '{cls.FINANCE.value}'")
        return cls.FINANCE


class ChromDBVectorStore:
    """Production-grade ChromaDB Vector Store wrapper with thread-safe persistence and diagnostics."""

    def __init__(self) -> None:
        self.db_path = settings.CHROMA_DB_PATH
        self.client = chromadb.PersistentClient(path=self.db_path)
        self._write_lock = threading.Lock()
        self._collections_cache: Dict[str, Collection] = {}
        self._hnsw_config = {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:M": 32,
            "hnsw:search_ef": 100,
        }
        self.setup_collections()

    def check_status(self) -> Dict[str, Any]:
        """Perform a vector store health check and return connection diagnostics."""
        try:
            heartbeat = self.client.heartbeat()
            cached_count = len(self._collections_cache)
            return {
                "healthy": True,
                "heartbeat": heartbeat,
                "db_path": self.db_path,
                "initialized_collections": cached_count,
            }
        except Exception as e:
            logger.error(f"ChromaDB at {self.db_path} is unresponsive: {e}", exc_info=True)
            return {
                "healthy": False,
                "error": str(e),
                "db_path": self.db_path,
            }

    def setup_collections(self) -> Dict[str, Collection]:
        """Idempotently configure and cache all canonical collections."""
        with self._write_lock:
            try:
                # Include configured collections plus canonical FinancialCollection enums
                configured_names = set(settings.COLLECTION_NAMES)
                canonical_names = {col.value for col in FinancialCollection}
                all_target_collections = sorted(configured_names.union(canonical_names))

                for name in all_target_collections:
                    collection = self.client.get_or_create_collection(
                        name=name,
                        metadata=self._hnsw_config
                    )
                    self._collections_cache[name] = collection

                logger.info(f"ChromaDB initialized {len(self._collections_cache)} collections successfully.")
                return self._collections_cache
            except Exception as e:
                logger.error(f"Failed to setup ChromaDB collections: {e}", exc_info=True)
                raise e

    def get_collection(self, collection_name: str | FinancialCollection) -> Collection:
        """Retrieve a collection instance by name or enum, creating it if needed."""
        name = collection_name.value if isinstance(collection_name, FinancialCollection) else str(collection_name)
        if name in self._collections_cache:
            return self._collections_cache[name]

        with self._write_lock:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata=self._hnsw_config
            )
            self._collections_cache[name] = collection
            return collection

    def get_collection_stats(self) -> List[Dict[str, Any]]:
        """Return document chunk counts and status for all active collections."""
        stats = []
        for name, col in self._collections_cache.items():
            try:
                count = col.count()
                stats.append({
                    "collection": name,
                    "chunk_count": count,
                    "status": "ready"
                })
            except Exception as e:
                logger.warning(f"Failed to fetch count for collection '{name}': {e}")
                stats.append({
                    "collection": name,
                    "chunk_count": 0,
                    "status": "error",
                    "error": str(e)
                })
        return stats

    def delete_document_chunks(self, collection_name: str, file_id: str) -> int:
        """Delete all chunks associated with a specific file_id in a thread-safe manner."""
        with self._write_lock:
            collection = self.get_collection(collection_name)
            existing = collection.get(where={"file_id": file_id})
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for file_id='{file_id}' in '{collection_name}'")
            return len(ids_to_delete)

    def delete_by_file_hash(self, collection_name: str, file_hash: str) -> int:
        """Delete any existing document chunks with matching file_hash for idempotent re-ingestion."""
        with self._write_lock:
            collection = self.get_collection(collection_name)
            existing = collection.get(where={"file_hash": file_hash})
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Overwritten {len(ids_to_delete)} stale chunks with hash '{file_hash}' in '{collection_name}'")
            return len(ids_to_delete)

    def clear_collection(self, collection_name: str) -> None:
        """Purge all documents from a specific collection and reset it."""
        with self._write_lock:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Purged collection '{collection_name}' from ChromaDB.")
            except Exception as e:
                logger.warning(f"Could not delete collection '{collection_name}': {e}")
            # Recreate empty collection
            created = self.client.get_or_create_collection(
                name=collection_name,
                metadata=self._hnsw_config
            )
            self._collections_cache[collection_name] = created


# Singleton instance
vector_store = ChromDBVectorStore()