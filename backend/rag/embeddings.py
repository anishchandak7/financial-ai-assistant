"""Embeddings Layer with Chunk Batching, Tenacity Retries, and Task-Type Specialization.

Uses Google Generative AI embeddings (gemini-embedding-2-preview) with 768 dimensions,
chunked batching to avoid API payload limits, and exponential backoff retry.
"""

import asyncio
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)


class EmbeddingLayer:
    """Enterprise embeddings manager providing batched, resilient embedding generation."""

    def __init__(self, embedding_service: str = "google", batch_size: int = 50) -> None:
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self._doc_embeddings = None
        self._query_embeddings = None

    def get_embedding_function(self, task_type: str = "retrieval_document") -> GoogleGenerativeAIEmbeddings:
        """Return the GoogleGenerativeAIEmbeddings instance tailored for documents or queries."""
        if task_type == "retrieval_query":
            if self._query_embeddings is None:
                self._query_embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2-preview",
                    api_key=settings.GEMINI_KEY,
                    output_dimensionality=768,
                    task_type="retrieval_query"
                )
            return self._query_embeddings

        if self._doc_embeddings is None:
            self._doc_embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                api_key=settings.GEMINI_KEY,
                output_dimensionality=768,
                task_type="retrieval_document"
            )
        return self._doc_embeddings

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _embed_batch_with_retry(self, embeddings_fn: GoogleGenerativeAIEmbeddings, batch: List[str]) -> List[List[float]]:
        """Asynchronously embed a single chunk batch with exponential backoff on transient errors."""
        return await embeddings_fn.aembed_documents(texts=batch)

    async def create_embeddings(self, documents: List[str]) -> List[List[float]]:
        """Batch and embed a list of document chunks asynchronously with error resilience."""
        if not documents:
            return []

        try:
            embeddings_fn = self.get_embedding_function(task_type="retrieval_document")
            total_chunks = len(documents)
            all_vectors: List[List[float]] = []

            for i in range(0, total_chunks, self.batch_size):
                batch = documents[i : i + self.batch_size]
                batch_vectors = await self._embed_batch_with_retry(embeddings_fn, batch)
                all_vectors.extend(batch_vectors)

            logger.info(
                f"Generated embeddings for {total_chunks} chunks in "
                f"{(total_chunks + self.batch_size - 1) // self.batch_size} batches. "
                f"Vector dim: {len(all_vectors[0]) if all_vectors else 0}."
            )
            return all_vectors

        except Exception as e:
            logger.error(f"Failed to create document embeddings: {e}", exc_info=True)
            raise e

    async def create_query_embedding(self, query: str) -> List[float]:
        """Generate a query embedding vector optimized with retrieval_query task type."""
        try:
            embeddings_fn = self.get_embedding_function(task_type="retrieval_query")
            vector = await embeddings_fn.aembed_query(text=query)
            return vector
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}", exc_info=True)
            raise e


# Singleton instance
embedding_layer = EmbeddingLayer()