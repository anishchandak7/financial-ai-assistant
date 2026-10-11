import uuid
from typing import Optional
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
from observability.logger import get_custom_logger
from rag.vector_store import vector_store
from rag.embeddings import embedding_layer

logger = get_custom_logger(name=__name__)

class RAGService:
    """Handles PDF parsing, chunking, embedding generation, and vector store ingestion."""

    def __init__(self) -> None:
        self.embedding_layer = embedding_layer
        self.vector_store = vector_store
        self.collection_mapping = vector_store.setup_collections()

    async def ingest_file(
        self,
        file_name: str,
        file_bytes: bytes,
        file_type: str = "application/pdf",
        collection_name: str = "Finance",
        file_id: Optional[str] = None
    ):
        try:
            effective_file_id = file_id or str(uuid.uuid4())
            logger.info(f"Ingesting file '{file_name}' into collection '{collection_name}' (ID: {effective_file_id})")

            parser = PyPDFium2Parser(extract_images=False, mode="page")
            file_blob = Blob.from_data(data=file_bytes)

            documents = list(parser.lazy_parse(blob=file_blob))
            logger.info(f"Loaded {len(documents)} document pages.")

            chunk_ids = []
            metadatas = []
            texts = []

            for i, doc in enumerate(documents):
                unique_chunk_id = f"{effective_file_id}_{i}"
                chunk_ids.append(unique_chunk_id)
                texts.append(doc.page_content)

                raw_meta = doc.metadata.copy() if doc.metadata else {}
                raw_meta.update({
                    "chunk_id": unique_chunk_id,
                    "file_id": effective_file_id,
                    "file_type": file_type,
                    "file_name": file_name
                })

                clean_meta = {}
                for key, value in raw_meta.items():
                    if value is None:
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        clean_meta[key] = value
                    else:
                        clean_meta[key] = str(value)

                metadatas.append(clean_meta)

            embeddings = await self.embedding_layer.create_embeddings(documents=texts)

            if collection_name not in self.collection_mapping:
                self.collection_mapping = self.vector_store.setup_collections()

            self.collection_mapping[collection_name].add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Successfully ingested {len(texts)} chunks into collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to ingest file '{file_name}': {e}", exc_info=True)
            raise e

rag_service = RAGService()
