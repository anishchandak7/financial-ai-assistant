from core.logger import get_custom_logger
from core.config import settings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

logger = get_custom_logger(name=__name__)

class EmbeddingLayer:
    
    def __init__(self, embedding_service:str = "google") -> None:
        self.embedding_config = {
            "google": {
                "model": "gemini-embedding-2-preview"
            }
        }
        supported_providers = self.embedding_config.keys()
        if embedding_service not in supported_providers:
            raise KeyError(f"Embedding service not supported: {embedding_service}")
        self.embedding_service = embedding_service

    def get_embedding_function(self):
        """Return the shared embedding function used by both ingestion and retrieval."""
        if self.embedding_service == "google":
            config = self.embedding_config.get(self.embedding_service)
            return GoogleGenerativeAIEmbeddings(
                model=config.get("model"),
                api_key=settings.GEMINI_KEY,
                output_dimensionality=768
            )
        raise KeyError(f"Embedding service not supported: {self.embedding_service}")
    
    async def create_embeddings(self, documents:list):
        try:
            # Use the same embedding model everywhere so query vectors match stored vectors.
            embeddings = self.get_embedding_function()
            vectors = await embeddings.aembed_documents(texts=documents)
            logger.info(msg=f"Created document embeddings of dim: ({len(vectors)}, {len(vectors[0])}).")
            return vectors
        except Exception as e:
            logger.error(msg="Failed to create embeddings for documents.", exc_info=True, stack_info=True)
            raise e