from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.specialists.base import BaseSpecialistAgent
from rag.retriever import financial_retriever
from rag.vector_store import FinancialCollection
from core.config import settings
from observability.logger import get_custom_logger
from observability.tracer import trace_span

logger = get_custom_logger(name=__name__)


class Citation(BaseModel):
    source_index: int = Field(
        ...,
        description="The 0-based index of the source document used to answer the question."
    )
    quote: str = Field(
        ...,
        description="The exact, verbatim quote from the source document that supports the answer."
    )


class CitedResponse(BaseModel):
    response: str = Field(
        ...,
        description="The complete answer to the user question. Incorporate inline citations like [0] or [1] where appropriate."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="The list of supporting citations extracted from the sources."
    )


SYSTEM_PROMPT = """
You are a Senior Financial QA Assistant responsible for answering queries related to finance, financial statements, accounting principles, and economic concepts.

IMPORTANT: YOU ARE STRICTLY PROHIBITED FROM PROVIDING FIDUCIARY OR PERSONAL FINANCIAL ADVICE. IN CASE THE USER ASKS FOR PERSONAL RECOMMENDATIONS, POLITELY EXPLAIN THAT YOU CANNOT PROVIDE PERSONAL ADVICE AND CLARIFY THE GENERAL EDUCATIONAL CONCEPTS.

Address questions by synthesizing the context information provided below. Every time you use information from a source, you MUST create a citation containing the source index and the exact, verbatim quote.
Add inline citations like [0] or [1] in your answer text.

Sources:
{context}

Your response should be clear, professional, user-friendly, and well-structured. Create analogies with real-world examples where appropriate, and ensure accurate correlation with technical financial concepts. If the sources do not contain sufficient specific information, provide an educational explanation based on standard financial principles and clearly state what additional details could be verified.
"""


from langchain_core.prompts.chat import MessagesPlaceholder
from memory.manager import agent_memory_manager

class FinanceQAAgent(BaseSpecialistAgent):
    """RAG-grounded Financial QA agent retrieving from ChromaDB collections with multi-turn session memory."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY,
            temperature=0.0
        )

    @property
    def name(self) -> str:
        return "Finance Q&A Agent"

    def _stringify_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "No specific internal documents found in vector storage matching this query."
        context = []
        for i, doc in enumerate(docs):
            file_name = doc.metadata.get("file_name", f"Doc_{i}")
            page = doc.metadata.get("page", 1)
            context.append(
                f"Source [{i}] (File: {file_name}, Page: {page})\n"
                f"Content: {doc.page_content}\n"
                f"Metadata: {doc.metadata}\n---"
            )
        return "\n".join(context)

    def run(self, input_query: str, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        with trace_span("finance_qa.run", {"query": input_query[:80]}):
            try:
                logger.info(f"FinanceQAAgent executing query: {input_query[:80]} (Session: {session_id})")

                # Retrieve dialogue history for Finance QA Agent
                chat_history = []
                working_summary = ""
                if session_id:
                    chat_history = agent_memory_manager.get_messages_for_agent(session_id, self.name, max_turns=6)
                    working_summary = agent_memory_manager.get_working_memory_summary(session_id, self.name)

                # Use Unified Financial Retriever with Self-Querying and RRF
                retrieved_docs: List[Document] = financial_retriever.retrieve_sync(
                    query=input_query,
                    collection_name=FinancialCollection.FINANCE,
                    top_k=5,
                    enable_multi_query=True,
                    enable_self_query=True,
                    min_relevance_threshold=0.25
                )

                context_str = self._stringify_docs(retrieved_docs)
                if working_summary:
                    context_str = f"{working_summary}\n\n{context_str}"

                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}")
                ])

                llm_with_citations = self.llm.with_structured_output(CitedResponse)
                rag_chain = prompt | llm_with_citations

                result: CitedResponse = rag_chain.invoke({
                    "context": context_str,
                    "chat_history": chat_history,
                    "query": input_query
                })

                citations_data = []
                if result.citations and retrieved_docs:
                    for cit in result.citations:
                        cit_info = {
                            "source_index": cit.source_index,
                            "quote": cit.quote,
                            "file_name": f"Source [{cit.source_index}]",
                            "metadata": {}
                        }
                        if 0 <= cit.source_index < len(retrieved_docs):
                            meta = retrieved_docs[cit.source_index].metadata or {}
                            cit_info["file_name"] = meta.get("file_name", f"Source [{cit.source_index}]")
                            cit_info["metadata"] = meta
                        citations_data.append(cit_info)

                # Persist turn in Finance Q&A Agent's memory partition
                if session_id:
                    agent_memory_manager.add_turn(
                        session_id=session_id,
                        agent_name=self.name,
                        user_query=input_query,
                        agent_response=result.response,
                        metadata={"citations_count": len(citations_data)}
                    )
                    if citations_data:
                        agent_memory_manager.update_working_memory(
                            session_id=session_id,
                            agent_name=self.name,
                            key="last_cited_sources",
                            value=[c.get("file_name") for c in citations_data]
                        )

                return {
                    "response": result.response,
                    "citations": citations_data,
                    "retrieved_contexts": [d.page_content for d in retrieved_docs] if retrieved_docs else []
                }

            except Exception as e:
                logger.error(f"Error in FinanceQAAgent: {e}", exc_info=True)
                return {
                    "response": f"An error occurred while answering your finance query: {str(e)}",
                    "citations": []
                }
