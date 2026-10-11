"""Automated Unit & Integration Tests for Production RAG Implementation."""

import unittest
import asyncio
from rag.vector_store import vector_store, FinancialCollection
from rag.retriever import financial_retriever, StructuredQueryFilter
from services.rag_service import rag_service


class TestVectorStoreAndCollections(unittest.TestCase):
    """Test ChromaDB vector store initialization, collections, and stats."""

    def test_canonical_collections_enum(self):
        """Verify FinancialCollection enum resolves variations properly."""
        self.assertEqual(FinancialCollection.from_str("Finance"), FinancialCollection.FINANCE)
        self.assertEqual(FinancialCollection.from_str("market_analysis"), FinancialCollection.MARKET_ANALYSIS)
        self.assertEqual(FinancialCollection.from_str("PORTFOLIO"), FinancialCollection.PORTFOLIO)
        self.assertEqual(FinancialCollection.from_str("unknown_xyz"), FinancialCollection.FINANCE)

    def test_vector_store_health_and_setup(self):
        """Verify vector store heartbeat and collections cache."""
        status = vector_store.check_status()
        self.assertTrue(status.get("healthy"))
        self.assertGreaterEqual(status.get("initialized_collections", 0), 6)

    def test_collection_stats(self):
        """Verify collection stats reporting."""
        stats = vector_store.get_collection_stats()
        self.assertTrue(isinstance(stats, list))
        collection_names = [s["collection"] for s in stats]
        self.assertIn("Finance", collection_names)
        self.assertIn("MarketAnalysis", collection_names)


class TestRAGIngestionAndMetadata(unittest.TestCase):
    """Test multi-format parsing, chunking, enrichment, and idempotency."""

    def test_metadata_extraction(self):
        """Verify regex and rule-based financial metadata extraction."""
        meta = rag_service._extract_metadata_from_name_and_content(
            file_name="AAPL_2024_Q3_10-Q_Report.pdf",
            sample_text="Quarterly report for Apple Inc. for the fiscal quarter ended June 29, 2024."
        )
        self.assertEqual(meta["ticker"], "AAPL")
        self.assertEqual(meta["doc_type"], "10-Q")
        self.assertEqual(meta["fiscal_year"], 2024)
        self.assertEqual(meta["fiscal_quarter"], "Q3")

    def test_text_and_csv_parsing(self):
        """Verify text and CSV loaders produce valid chunkable documents."""
        # Text
        txt_bytes = b"Consolidated Balance Sheet. Total assets amounted to $352 billion."
        docs = rag_service._parse_raw_document(txt_bytes, "balance_sheet.txt", "text/plain")
        self.assertEqual(len(docs), 1)
        self.assertIn("Consolidated Balance Sheet", docs[0].page_content)

        # CSV
        csv_bytes = b"Ticker,Revenue,NetIncome\nNVDA,26044,14881\nMSFT,62020,21870"
        csv_docs = rag_service._parse_raw_document(csv_bytes, "financials.csv", "text/csv")
        self.assertEqual(len(csv_docs), 1)
        self.assertTrue(csv_docs[0].metadata.get("is_tabular"))
        self.assertIn("NVDA | 26044", csv_docs[0].page_content)

    def test_end_to_end_ingestion_and_idempotency(self):
        """Test document ingestion into ChromaDB and idempotent overwrite on duplicate hash."""
        async def _run():
            test_content = (
                "Company: NVIDIA Corp (NVDA). Fiscal Year 2024 Annual 10-K Report.\n"
                "Data Center revenue grew 217% year-over-year driven by accelerated computing platforms.\n"
                "Gross margin expanded to 72.7% compared to 56.9% in the prior fiscal year.\n"
                "Operating expenses totaled $11.3 billion."
            ).encode("utf-8")

            # 1. Ingest initial document
            res1 = await rag_service.ingest_file(
                file_name="NVDA_2024_10K_Annual.txt",
                file_bytes=test_content,
                file_type="text/plain",
                collection_name="MarketAnalysis"
            )
            self.assertEqual(res1["collection"], "MarketAnalysis")
            self.assertGreater(res1["chunk_count"], 0)
            initial_chunks = res1["chunk_count"]

            # 2. Re-ingest IDENTICAL content (idempotency check)
            res2 = await rag_service.ingest_file(
                file_name="NVDA_2024_10K_Annual.txt",
                file_bytes=test_content,
                file_type="text/plain",
                collection_name="MarketAnalysis"
            )
            self.assertEqual(res2["file_hash"], res1["file_hash"])
            self.assertEqual(res2["chunk_count"], initial_chunks)

            # 3. Retrieve via FinancialRetriever
            retrieved = financial_retriever.retrieve_sync(
                query="What was NVIDIA's gross margin in fiscal 2024?",
                collection_name="MarketAnalysis",
                top_k=3,
                enable_multi_query=False,
                enable_self_query=False
            )
            self.assertGreater(len(retrieved), 0)
            self.assertIn("gross margin", retrieved[0].page_content.lower())

            # 4. Clean up
            deleted = await rag_service.delete_document(res2["file_id"])
            self.assertGreater(deleted, 0)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
