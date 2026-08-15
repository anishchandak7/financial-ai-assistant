from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
from streamlit.runtime.uploaded_file_manager import UploadedFile
from core.logger import get_custom_logger
from rag.vector_store import vector_store
from rag.embeddings import EmbeddingLayer

logger = get_custom_logger(name=__name__)

class RAGService:
    
    def __init__(self) -> None:
        self.embedding_layer = EmbeddingLayer()
        self.vector_store = vector_store 
        self.collection_mapping = vector_store.setup_collections()
    
    async def ingest_file(self, file: UploadedFile, collection_name:str = "Finance"):
        try:
            logger.info(msg=f"Ingesting file {file.name} of type {file.type}")
            
            parser = PyPDFium2Parser(extract_images=True, mode='page')
            file_blob = Blob.from_data(data=file.getvalue())
            
            documents = list(parser.lazy_parse(blob=file_blob))
            logger.info(msg=f"Loaded {len(documents)} pages.")
            
            chunk_ids = []
            metadatas = []
            texts = []
            
            for i, doc in enumerate(documents):
                unique_chunk_id = f"{file.file_id}_{i}"
                chunk_ids.append(unique_chunk_id)
                texts.append(doc.page_content)
                
                # 1. Combine metadata
                raw_meta = doc.metadata.copy() if doc.metadata else {}
                raw_meta.update({
                    "chunk_id": unique_chunk_id,
                    "file_id": file.file_id,
                    "file_type": file.type,
                    "file_name": file.name
                })
                
                # 2. Sanitize metadata for ChromaDB compliance
                clean_meta = {}
                for key, value in raw_meta.items():
                    if value is None:
                        continue  # ChromaDB cannot accept None, so we skip it
                    elif isinstance(value, (str, int, float, bool)):
                        clean_meta[key] = value  # Keep supported types as-is
                    else:
                        clean_meta[key] = str(value)  # Cast lists/objects to strings
                
                metadatas.append(clean_meta)
                
            embeddings = await self.embedding_layer.create_embeddings(documents=texts)
            
            self.collection_mapping[collection_name].add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(msg=f"File ingested in the vector store at '{collection_name}' collection.")
            
        except Exception as e:
            logger.error(msg=f"Failed to ingest the file {file.name}.", exc_info=True, stack_info=True)
            raise e

rag_service = RAGService()