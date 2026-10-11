"""Enterprise RAG Ingestion and Document Lifecycle Service.

Features:
1. Multi-format parsing (PDF via PyPDFium2, UTF-8 Text/Markdown, CSV Tabular representation).
2. Financial Recursive Token/Character Chunking preserving section boundaries and page numbers.
3. Content Context Injection: Prepending document and page anchors to chunks for higher semantic recall.
4. Rich Financial Metadata Extraction (Ticker, Doc Type, Fiscal Year/Quarter, Timestamps).
5. SHA-256 Content-Hash Idempotency (Clean overwrite on re-upload, zero duplicate index bloat).
6. Non-blocking asynchronous offloading to threadpool.
7. Document lifecycle management (listing documents, inspecting chunk counts, deleting files).
"""

import asyncio
import csv
import datetime
import hashlib
import io
import re
import uuid
from typing import List, Dict, Any, Optional

from langchain_core.documents.base import Blob, Document
from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from observability.logger import get_custom_logger
from rag.vector_store import vector_store, FinancialCollection
from rag.embeddings import embedding_layer

logger = get_custom_logger(name=__name__)


class RAGService:
    """Manages document parsing, chunking, enrichment, embedding, and vector storage."""

    def __init__(self) -> None:
        self.embedding_layer = embedding_layer
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def _extract_metadata_from_name_and_content(self, file_name: str, sample_text: str) -> Dict[str, Any]:
        """Extract financial metadata (ticker, doc_type, fiscal year, fiscal quarter) from filename and text."""
        metadata: Dict[str, Any] = {
            "ticker": "GENERAL",
            "doc_type": "general",
            "fiscal_year": datetime.datetime.now(datetime.timezone.utc).year,
            "fiscal_quarter": "FY"
        }

        # 1. Detect Ticker Symbol (e.g. AAPL, NVDA, TSLA, MSFT)
        tokens = re.split(r"[\s_.\-]+", file_name)
        common_words = {"THE", "AND", "FOR", "INC", "CORP", "PDF", "TXT", "CSV", "DOC", "Q1", "Q2", "Q3", "Q4", "FY", "REPORT", "ANNUAL", "QUARTERLY"}
        for tok in tokens:
            if tok.isupper() and 2 <= len(tok) <= 5 and tok not in common_words and not tok.isdigit():
                metadata["ticker"] = tok
                break

        # 2. Detect Document Type
        name_lower = (file_name + " " + sample_text[:1000]).lower()
        if "10-k" in name_lower or "annual report" in name_lower:
            metadata["doc_type"] = "10-K"
        elif "10-q" in name_lower or "quarterly report" in name_lower:
            metadata["doc_type"] = "10-Q"
        elif "8-k" in name_lower:
            metadata["doc_type"] = "8-K"
        elif "earnings" in name_lower or "transcript" in name_lower or "call" in name_lower:
            metadata["doc_type"] = "earnings_transcript"
        elif "research" in name_lower or "analyst" in name_lower or "equity" in name_lower:
            metadata["doc_type"] = "analyst_research"
        elif "portfolio" in name_lower or "statement" in name_lower:
            metadata["doc_type"] = "portfolio_statement"

        # 3. Detect Fiscal Year
        year_match = re.search(r"(?:^|[^0-9])(20[1-3][0-9])(?:[^0-9]|$)", file_name + " " + sample_text[:500])
        if year_match:
            metadata["fiscal_year"] = int(year_match.group(1))

        # 4. Detect Fiscal Quarter
        quarter_match = re.search(r"(?:^|[^a-zA-Z0-9])(Q[1-4])(?:[^a-zA-Z0-9]|$)", file_name + " " + sample_text[:500], re.IGNORECASE)
        if quarter_match:
            metadata["fiscal_quarter"] = quarter_match.group(1).upper()

        return metadata

    def _parse_raw_document(self, file_bytes: bytes, file_name: str, file_type: str) -> List[Document]:
        """Multi-format parser supporting PDF, UTF-8 Text/Markdown, and CSV Tabular data."""
        fname_lower = file_name.lower()

        # A. CSV / Tabular Files
        if fname_lower.endswith(".csv") or "csv" in file_type.lower():
            try:
                decoded_str = file_bytes.decode("utf-8", errors="replace")
                reader = csv.reader(io.StringIO(decoded_str))
                rows = list(reader)
                if not rows:
                    return [Document(page_content="Empty CSV file", metadata={"page": 1, "is_tabular": True})]

                headers = rows[0]
                tabular_lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
                for r in rows[1:1000]:  # Cap at 1000 rows
                    tabular_lines.append(" | ".join(r))
                content = f"### Structured Financial Data Table: {file_name}\n\n" + "\n".join(tabular_lines)
                return [Document(page_content=content, metadata={"page": 1, "is_tabular": True})]
            except Exception as e:
                logger.warning(f"CSV parsing error, fallback to raw text: {e}")

        # B. Plain Text / Markdown
        if fname_lower.endswith((".txt", ".md", ".json")) or "text" in file_type.lower():
            content = file_bytes.decode("utf-8", errors="replace")
            return [Document(page_content=content, metadata={"page": 1, "is_tabular": False})]

        # C. PDF (Default)
        try:
            parser = PyPDFium2Parser(extract_images=False, mode="page")
            file_blob = Blob.from_data(data=file_bytes)
            pages = list(parser.lazy_parse(blob=file_blob))
            for i, p in enumerate(pages):
                if not p.metadata:
                    p.metadata = {}
                p.metadata["page"] = i + 1
                p.metadata["is_tabular"] = False
            return pages
        except Exception as e:
            logger.error(f"Failed to parse document '{file_name}' as PDF: {e}", exc_info=True)
            # Graceful fallback: attempt utf-8 text extraction
            fallback_text = file_bytes.decode("utf-8", errors="ignore")
            if fallback_text.strip():
                return [Document(page_content=fallback_text, metadata={"page": 1, "is_tabular": False})]
            raise ValueError(f"Unable to parse document '{file_name}'. Ensure it is a valid PDF, TXT, or CSV file.")

    async def ingest_file(
        self,
        file_name: str,
        file_bytes: bytes,
        file_type: str = "application/pdf",
        collection_name: str = "Finance",
        file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """End-to-end idempotent ingestion pipeline with chunking, enrichment, and batched embeddings."""
        try:
            canonical_col = FinancialCollection.from_str(collection_name).value
            effective_file_id = file_id or str(uuid.uuid4())
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            upload_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

            logger.info(
                f"Starting ingestion for '{file_name}' ({len(file_bytes)} bytes) "
                f"into collection '{canonical_col}' (Hash: {file_hash[:8]})"
            )

            # 1. Parse Document in Threadpool
            raw_docs = await asyncio.to_thread(self._parse_raw_document, file_bytes, file_name, file_type)
            total_pages = len(raw_docs)
            sample_content = raw_docs[0].page_content if raw_docs else ""

            # 2. Extract Rich Financial Metadata
            financial_meta = self._extract_metadata_from_name_and_content(file_name, sample_content)

            # 3. Financial Recursive Chunking
            chunk_docs: List[Document] = []
            for doc in raw_docs:
                page_num = doc.metadata.get("page", 1)
                is_tab = doc.metadata.get("is_tabular", False)
                page_chunks = self.text_splitter.split_text(doc.page_content)

                for chunk_text in page_chunks:
                    clean_chunk = chunk_text.strip()
                    if not clean_chunk:
                        continue
                    # Anchor chunk with source header for contextual embedding
                    anchored_content = (
                        f"[Document: {file_name} | Page: {page_num} | Entity: {financial_meta['ticker']}]\n"
                        f"{clean_chunk}"
                    )
                    chunk_docs.append(
                        Document(
                            page_content=anchored_content,
                            metadata={
                                "page": page_num,
                                "is_tabular": is_tab
                            }
                        )
                    )

            if not chunk_docs:
                raise ValueError(f"No extractable text content found in document '{file_name}'.")

            total_chunks = len(chunk_docs)
            logger.info(f"Generated {total_chunks} contextual chunks from {total_pages} pages.")

            # 4. Prepare ChromaDB Payload
            chunk_ids = []
            texts = []
            metadatas = []

            for idx, ch in enumerate(chunk_docs):
                unique_chunk_id = f"{effective_file_id}_{idx}"
                chunk_ids.append(unique_chunk_id)
                texts.append(ch.page_content)

                chunk_meta = {
                    "chunk_id": unique_chunk_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "file_id": effective_file_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_hash": file_hash,
                    "page": ch.metadata.get("page", 1),
                    "is_tabular": bool(ch.metadata.get("is_tabular", False)),
                    "ticker": str(financial_meta.get("ticker", "GENERAL")),
                    "doc_type": str(financial_meta.get("doc_type", "general")),
                    "fiscal_year": int(financial_meta.get("fiscal_year", 2024)),
                    "fiscal_quarter": str(financial_meta.get("fiscal_quarter", "FY")),
                    "upload_timestamp": upload_time,
                }
                metadatas.append(chunk_meta)

            # 5. Generate Embeddings (Batched with Tenacity Retries)
            embeddings = await self.embedding_layer.create_embeddings(documents=texts)

            # 6. Thread-Safe Idempotent ChromaDB Write (Overwriting Stale Chunks with Same Hash)
            def _write_to_chroma():
                # Overwrite duplicate chunks if exact file already indexed
                self.vector_store.delete_by_file_hash(canonical_col, file_hash)
                collection = self.vector_store.get_collection(canonical_col)
                collection.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )

            await asyncio.to_thread(_write_to_chroma)
            logger.info(f"Successfully ingested '{file_name}' ({total_chunks} chunks) into '{canonical_col}'.")

            return {
                "file_id": effective_file_id,
                "file_name": file_name,
                "file_hash": file_hash,
                "collection": canonical_col,
                "chunk_count": total_chunks,
                "total_pages": total_pages,
                "metadata": financial_meta
            }

        except Exception as e:
            logger.error(f"Failed to ingest file '{file_name}': {e}", exc_info=True)
            raise e

    async def list_documents(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scan active collections and aggregate unique documents with metadata and chunk counts."""
        def _fetch_docs():
            collections_to_scan = (
                [FinancialCollection.from_str(collection_name).value]
                if collection_name
                else list(self.vector_store._collections_cache.keys())
            )
            doc_map: Dict[str, Dict[str, Any]] = {}

            for col_name in collections_to_scan:
                try:
                    col = self.vector_store.get_collection(col_name)
                    # Fetch stored metadata records
                    results = col.get(include=["metadatas"])
                    metadatas = results.get("metadatas") or []

                    for meta in metadatas:
                        if not meta:
                            continue
                        fid = meta.get("file_id")
                        if not fid:
                            continue

                        if fid not in doc_map:
                            doc_map[fid] = {
                                "file_id": fid,
                                "file_name": meta.get("file_name", "Unknown"),
                                "file_type": meta.get("file_type", "application/pdf"),
                                "file_hash": meta.get("file_hash", ""),
                                "collection": col_name,
                                "chunk_count": 0,
                                "total_pages": meta.get("page", 1),
                                "ticker": meta.get("ticker"),
                                "doc_type": meta.get("doc_type"),
                                "fiscal_year": meta.get("fiscal_year"),
                                "fiscal_quarter": meta.get("fiscal_quarter"),
                                "upload_timestamp": meta.get("upload_timestamp"),
                            }
                        doc_map[fid]["chunk_count"] += 1
                        if meta.get("page", 1) > doc_map[fid]["total_pages"]:
                            doc_map[fid]["total_pages"] = meta.get("page", 1)

                except Exception as e:
                    logger.warning(f"Error querying documents from collection '{col_name}': {e}")

            return list(doc_map.values())

        return await asyncio.to_thread(_fetch_docs)

    async def delete_document(self, file_id: str) -> int:
        """Find and remove all chunks belonging to file_id across collections."""
        def _delete():
            total_deleted = 0
            for col_name in list(self.vector_store._collections_cache.keys()):
                deleted = self.vector_store.delete_document_chunks(col_name, file_id)
                total_deleted += deleted
            return total_deleted

        return await asyncio.to_thread(_delete)


# Singleton instance
rag_service = RAGService()
