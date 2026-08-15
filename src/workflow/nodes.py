from pydantic import Field, BaseModel
from typing import List
from typing_extensions import Literal
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from workflow.state import RouterState
from rag.vector_store import vector_store
from rag.embeddings import EmbeddingLayer
from core.config import settings
from core.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

class RouterResponse(BaseModel):
    steps: List[Literal["Finance Q&A Agent", "Portfolio Analysis Agent", "Market Analysis Agent", "Goal Planning Agent","News Synthesizer Agent", "Tax Education Agent"]] = Field(None, description="The next step in the routing process")

class Citation(BaseModel):
    source_index: int = Field(
        ...,
        description="The 0-based index of the source document used to answer the question."
    )
    quote:str = Field(
        ...,
        description="The exact, verbatim quote from the source document that supports the answer."
    )

class CitedResponse(BaseModel):
    response: str = Field(
        ...,
        description="The complete answer to the user question. Incorporate inline citations like [0] or [1] where appropriate."
    )
    citations: List[Citation] = Field(
        ...,
        description="The list of supporting citations extracted from the sources."
    )

def llm_based_router(state: RouterState):
    """Route the input to the appropriate Agent."""
    input_query = state.get("input_query")
    model_name = state.get("model_name")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=settings.GEMINI_KEY
    )
    
    router = llm.with_structured_output(RouterResponse)
    
    SYSTEM_PROMPT = """
        You are Routing Agent with access to following Agents.
        Routing decision should be based on Agent capabilities.

        Agent: Finance Q&A Agent
        Capabilities: It handles the finance related question and answers.

        Agent: Portfolio Analysis Agent
        Capabilities: It handles queries related to portfolio analysis of the user.

        Agent: Market Analysis Agent
        Capabilities: It handles queries related to stock market analysis and provide realtime market insights.

        Agent: Goal Planning Agent
        Capabilities: It handles queries related to assist user with financial goal setting and planning.

        Agent: News Synthesizer Agent
        Capabilities: It handles the queries related to financial news.

        Agent: Tax Education Agent
        Capabilities: It handles the queries related to tax concepts and account types.
    """
    logger.info(msg="Workflow Router")
    logger.info(msg=f"Invoking router with query: {input_query}")
    decision = router.invoke(
            input=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=input_query)
            ]
        )
    logger.info(msg=f"Router decision: {decision.steps}")
    steps = list(decision.steps) if decision and decision.steps else []
    state["decision"] = steps
    state["routing_decisions"] = list(steps)
    return state

def finance_qa_node(state: RouterState):
    logger.info(msg="Finance Q&A Node")
    
    input_query = state.get("input_query")
    model_name = state.get("model_name")
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=settings.GEMINI_KEY
    )
    
    collection = settings.COLLECTION_NAMES[0] # Finance

    # Share the same embedding model for query-time matching.
    # MMR needs an embedding function on the Chroma object before it can search.
    embedding_layer = EmbeddingLayer()
    retriever = Chroma(
        collection_name=collection,
        client=vector_store.client,
        embedding_function=embedding_layer.get_embedding_function()
    ).as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
    )
    
    def stringify_docs_with_source_and_metadata(docs: List[Document]) -> str:
        context = []
        for i, doc in enumerate(docs):
            context.append(
                f"Source [{i}]\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n---"
            )
        return "\n".join(context)
    
    llm_with_citations = llm.with_structured_output(CitedResponse)
    
    SYSTEM_PROMPT = """
    You are a QA Assistant responsible to answer only queries related to Finance.
    
    IMPORTANT: YOU ARE NOT ALLOWED TO PROVIDE ANY KIND OF FINANCIAL ADVISE. IN CASE
    USER TRIES TO TRICK YOU ASKING FINANCIAL ADVISES, POLITELY RESPOND YOU ARE CANNOT
    ADDRESS THE QUERY.
    
    Address questions related finance such as asking about certain concept, topic, details
    by referring below context information. Every time you use information from a source, 
    you MUST create a citation containing the source index and the exact quote. 
    Add inline citations like [0] or [1] in your answer text.\n\n
    Sources:
    {context}
    
    Your response should user friendly and easy to understand. Try to create analogy with 
    real world examples, if questions are about asking any finance concept/topic/details
    and ensure to provided proper correlation with technical information. 
    """
    
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", SYSTEM_PROMPT),
            ("human", "{query}")
        ]
    )
    
    retrieved_docs: List[Document] = retriever.invoke(input=input_query)
    context_str = stringify_docs_with_source_and_metadata(retrieved_docs)
    
    rag_chain = prompt | llm_with_citations
    result: CitedResponse = rag_chain.invoke({
        "context": context_str,
        "query": input_query
    })
    
    logger.info(msg=f"Query: {input_query}")
    logger.info(msg=f"Response: {result.response}")
    
    citations_data = []
    if result.citations:
        logger.info(msg="Citations:")
        for cit in result.citations:
            logger.info(msg=f"- Source index: [{cit.source_index}] | Exact Quote: [{cit.quote}]")
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
    
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
        
    state["agent_output"]["Finance Q&A Agent"] = {
        "response": result.response,
        "citations": citations_data
    }
    state["citations"] = citations_data
    return state

def portfolio_analysis_node(state: RouterState):
    logger.info(msg="Portfolio Analysis Node")
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"]["Portfolio Analysis Agent"] = {
        "response": "Portfolio Analysis feature is currently under active development. Please stay tuned!"
    }
    return state

def market_analysis_node(state: RouterState):
    logger.info(msg="Market Analysis Node")
    input_query = state.get("input_query")
    model_name = state.get("model_name")
    
    from agents.market_analysis_agent import MarketAnalysisAgent
    agent = MarketAnalysisAgent(model_name=model_name)
    result = agent.run(input_query=input_query)
    
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
        
    state["agent_output"]["Market Analysis Agent"] = result
    
    if result.get("citations"):
        if "citations" not in state or state["citations"] is None:
            state["citations"] = []
        state["citations"].extend(result["citations"])
        
    return state

def goal_planning_node(state: RouterState):
    logger.info(msg="Goal Planning Node")
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"]["Goal Planning Agent"] = {
        "response": "Goal Planning feature is currently under active development. Please stay tuned!"
    }
    return state

def news_synthesizer_node(state: RouterState):
    logger.info(msg="News Synthesizer Node")
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"]["News Synthesizer Agent"] = {
        "response": "News Synthesizer feature is currently under active development. Please stay tuned!"
    }
    return state

def tax_education_node(state: RouterState):
    logger.info(msg="Tax Education Node")
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"]["Tax Education Agent"] = {
        "response": "Tax Education feature is currently under active development. Please stay tuned!"
    }
    return state