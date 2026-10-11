from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.specialists.base import BaseSpecialistAgent
from rag.vector_store import vector_store
from rag.embeddings import embedding_layer
from core.config import settings
from observability.logger import get_custom_logger

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
You are a QA Assistant responsible to answer queries related to Finance and financial concepts.

IMPORTANT: YOU ARE NOT ALLOWED TO PROVIDE ANY KIND OF FIDUCIARY OR PERSONAL FINANCIAL ADVICE. IN CASE
THE USER ASKS FOR PERSONAL ADVICE, POLITELY RESPOND THAT YOU CANNOT ADDRESS PERSONAL ADVISORY REQUESTS.

Address questions related to finance such as concepts, accounting principles, and financial details
by referring to the context information provided below. Every time you use information from a source, 
you MUST create a citation containing the source index and the exact quote. 
Add inline citations like [0] or [1] in your answer text.

Sources:
{context}

Your response should be user-friendly and easy to understand. Create analogies with real-world examples
where appropriate, and ensure accurate correlation with technical financial concepts.
"""

class FinanceQAAgent(BaseSpecialistAgent):
    """RAG-grounded Financial QA agent retrieving from ChromaDB collections."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY
        )

    @property
    def name(self) -> str:
        return "Finance Q&A Agent"

    def _stringify_docs(self, docs: List[Document]) -> str:
        context = []
        for i, doc in enumerate(docs):
            context.append(
                f"Source [{i}]\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n---"
            )
        return "\n".join(context)

    def run(self, input_query: str, **kwargs) -> Dict[str, Any]:
        try:
            logger.info(f"FinanceQAAgent executing query: {input_query[:80]}")
            collection = settings.COLLECTION_NAMES[0]  # Finance

            retriever = Chroma(
                collection_name=collection,
                client=vector_store.client,
                embedding_function=embedding_layer.get_embedding_function()
            ).as_retriever(
                search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
            )

            retrieved_docs: List[Document] = retriever.invoke(input=input_query)
            context_str = self._stringify_docs(retrieved_docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{query}")
            ])

            llm_with_citations = self.llm.with_structured_output(CitedResponse)
            rag_chain = prompt | llm_with_citations

            result: CitedResponse = rag_chain.invoke({
                "context": context_str,
                "query": input_query
            })

            citations_data = []
            if result.citations:
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

            return {
                "response": result.response,
                "citations": citations_data
            }

        except Exception as e:
            logger.error(f"Error in FinanceQAAgent: {e}", exc_info=True)
            return {
                "response": f"An error occurred while answering your finance query: {str(e)}",
                "citations": []
            }
