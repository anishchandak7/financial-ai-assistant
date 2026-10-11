"""Enterprise Financial Retriever with Self-Querying, Multi-Query Expansion, and Reciprocal Rank Fusion (RRF).

Combines:
1. LLM-based Self-Querying: Extracts semantic search terms and structured metadata filters (ticker, year, doc_type).
2. Multi-Query Expansion: Generates diverse perspective queries to maximize recall.
3. Reciprocal Rank Fusion (RRF): Blends and re-ranks retrieved chunks across query variations.
4. Dynamic Collection Routing: Type-safe dispatch to canonical collections.
5. Relevance Thresholding: Prunes low-similarity background noise.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from rag.vector_store import vector_store, FinancialCollection
from rag.embeddings import embedding_layer
from core.config import settings
from observability.logger import get_custom_logger
from observability.tracer import trace_span

logger = get_custom_logger(name=__name__)


class StructuredQueryFilter(BaseModel):
    """Extracted semantic search terms and structured metadata filters from a natural language query."""
    semantic_query: str = Field(
        ...,
        description="The focused core question or concept for dense vector retrieval, stripped of specific metadata keywords."
    )
    ticker: Optional[str] = Field(
        None,
        description="Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA', 'MSFT') if specifically targeted in query."
    )
    fiscal_year: Optional[int] = Field(
        None,
        description="Specific 4-digit fiscal year (e.g. 2023, 2024, 2025) if specified in query."
    )
    doc_type: Optional[str] = Field(
        None,
        description="Document type if targeted: '10-K', '10-Q', 'earnings_transcript', 'analyst_research', 'portfolio_statement'."
    )


SELF_QUERY_SYSTEM_PROMPT = """You are a financial query analyzer for a ChromaDB document repository.
Your task is to parse a user's question into:
1. `semantic_query`: Clean core query keywords for cosine embedding search.
2. `ticker`: Ticker symbol if the question targets a specific stock (e.g. AAPL, NVDA, TSLA).
3. `fiscal_year`: Fiscal year as an integer (e.g. 2024) if specifically mentioned.
4. `doc_type`: Target document type if specifically asked for (e.g. 10-K, 10-Q, earnings_transcript).

If a metadata field is not mentioned, leave it as null."""


class FinancialRetriever:
    """Unified retrieval engine supporting self-querying, multi-query expansion, and RRF."""

    def __init__(self) -> None:
        self.vector_store = vector_store
        self.embedding_layer = embedding_layer
        self.model_name = settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY,
            temperature=0.0
        )

    def extract_structured_filter(self, query: str) -> StructuredQueryFilter:
        """Use structured LLM output to parse query into semantic text + metadata filter."""
        try:
            llm = self._get_llm()
            structured_extractor = llm.with_structured_output(StructuredQueryFilter)
            result = structured_extractor.invoke([
                SystemMessage(content=SELF_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=query)
            ])
            return result
        except Exception as e:
            logger.warning(f"Self-query extraction failed, falling back to raw query: {e}")
            return StructuredQueryFilter(semantic_query=query)

    def _build_where_clause(self, parsed: StructuredQueryFilter) -> Optional[Dict[str, Any]]:
        """Convert extracted metadata into ChromaDB-compatible `where` filter."""
        conditions = []
        if parsed.ticker and parsed.ticker.upper() != "NONE":
            conditions.append({"ticker": parsed.ticker.upper()})
        if parsed.fiscal_year:
            conditions.append({"fiscal_year": parsed.fiscal_year})
        if parsed.doc_type:
            conditions.append({"doc_type": parsed.doc_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def generate_query_variations(self, query: str, count: int = 3) -> List[str]:
        """Generate multi-query perspective variations for high recall."""
        prompt = (
            f"You are a financial research assistant. Generate {count} distinct search query variations "
            f"(different financial terminology, synonyms, or related concepts) to maximize retrieval recall.\n"
            f"Output exactly {count} queries, one per line. No bullets, numbers, or introductory text.\n\n"
            f"Original Query: {query}"
        )
        try:
            llm = self._get_llm()
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            variations = [line.strip() for line in text.strip().split("\n") if line.strip() and not line.startswith("[{")]
            return [query] + variations[:count]
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            return [query]

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        for doc_list in ranked_lists:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.metadata.get("chunk_id") or str(hash(doc.page_content))
                doc_store[doc_id] = doc
                # Standard RRF formula
                score = 1.0 / (k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # Sort by descending RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
        return [doc_store[did] for did in sorted_ids]

    def retrieve_sync(
        self,
        query: str,
        collection_name: str | FinancialCollection = FinancialCollection.FINANCE,
        top_k: int = 5,
        enable_multi_query: bool = True,
        enable_self_query: bool = True,
        min_relevance_threshold: float = 0.25
    ) -> List[Document]:
        """Synchronously execute retrieval with self-querying, multi-query expansion, and RRF."""
        canonical_col = FinancialCollection.from_str(str(collection_name)).value

        with trace_span("retriever.search", {"query": query[:80], "collection": canonical_col}):
            logger.info(f"Retrieving from '{canonical_col}' for query: '{query}'")

            # 1. Parse Self-Query Metadata Filter
            where_clause = None
            search_query = query
            if enable_self_query:
                parsed = self.extract_structured_filter(query)
                search_query = parsed.semantic_query or query
                where_clause = self._build_where_clause(parsed)
                logger.info(f"Parsed query: '{search_query}', where filter: {where_clause}")

            # 2. Multi-Query Variations
            queries = self.generate_query_variations(search_query, count=2) if enable_multi_query else [search_query]

            collection = self.vector_store.get_collection(canonical_col)
            ranked_runs: List[List[Document]] = []

            for q_var in queries:
                try:
                    # Query Chroma directly with embedding function
                    emb_fn = self.embedding_layer.get_embedding_function(task_type="retrieval_query")
                    query_vector = emb_fn.embed_query(q_var)

                    # First try with metadata filter
                    query_kwargs = {
                        "query_embeddings": [query_vector],
                        "n_results": min(top_k * 2, 10),
                        "include": ["documents", "metadatas", "distances"]
                    }
                    if where_clause:
                        query_kwargs["where"] = where_clause

                    results = collection.query(**query_kwargs)

                    # Fallback relaxation if strict metadata filter yielded 0 results
                    if where_clause and not (results.get("documents") and results["documents"][0]):
                        logger.info(f"Zero results with strict filter {where_clause}. Relaxing filter.")
                        query_kwargs.pop("where", None)
                        results = collection.query(**query_kwargs)

                    run_docs: List[Document] = []
                    docs = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0]

                    for d_text, d_meta, d_dist in zip(docs, metas, distances):
                        # For cosine space: similarity = 1 - cosine_distance
                        cosine_sim = 1.0 - (d_dist if d_dist is not None else 1.0)
                        if cosine_sim < min_relevance_threshold:
                            continue

                        meta = dict(d_meta) if d_meta else {}
                        meta["relevance_score"] = round(cosine_sim, 4)
                        run_docs.append(Document(page_content=d_text, metadata=meta))

                    ranked_runs.append(run_docs)

                except Exception as sub_err:
                    logger.warning(f"Error executing sub-query '{q_var}': {sub_err}")

            # 3. Fuse Results via Reciprocal Rank Fusion
            if enable_multi_query and len(ranked_runs) > 1:
                fused = self._reciprocal_rank_fusion(ranked_runs)
            elif ranked_runs:
                fused = ranked_runs[0]
            else:
                fused = []

            final_docs = fused[:top_k]
            logger.info(f"Retrieved {len(final_docs)} fused documents from collection '{canonical_col}'.")
            return final_docs

    async def retrieve(
        self,
        query: str,
        collection_name: str | FinancialCollection = FinancialCollection.FINANCE,
        top_k: int = 5,
        enable_multi_query: bool = True,
        enable_self_query: bool = True,
        min_relevance_threshold: float = 0.25
    ) -> List[Document]:
        """Asynchronously execute retrieval by offloading to threadpool."""
        return await asyncio.to_thread(
            self.retrieve_sync,
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            enable_multi_query=enable_multi_query,
            enable_self_query=enable_self_query,
            min_relevance_threshold=min_relevance_threshold
        )


# Singleton instance
financial_retriever = FinancialRetriever()
