"""LangGraph Orchestrator Graph.

Defines the multi-agent topology using LangGraph StateGraph:
1. Canonical Input Guardrail Node (LCEL input_guardrail_runnable) intercepting prompt injection / jailbreaks.
2. Triage & Dispatch Node (adaptive LLM-based intent routing).
3. Specialist Execution Nodes (Conversational, Market Analysis, Stock Technicals, RAG QA).
4. Canonical Output Compliance Node (LCEL compliance_runnable) appending regulatory disclosures.
"""

from langgraph.graph import StateGraph, START, END
from agents.orchestrator.state import GraphState
from agents.orchestrator.router import triage_and_route
from agents.specialists.conversational import ConversationalAgent
from agents.specialists.market_analyst import MarketAnalysisAgent
from agents.specialists.stock_analyst import StockAnalyzerAgent
from agents.specialists.finance_qa import FinanceQAAgent
from agents.guardrails.prompt_guard import input_guardrail_runnable, GuardrailVerdict
from agents.guardrails.compliance import compliance_runnable
from observability.logger import get_custom_logger
from observability.tracer import trace_span

logger = get_custom_logger(name=__name__)


# 1. Guardrail Input Node (LangChain LCEL Runnable)
def input_guardrail_node(state: GraphState) -> GraphState:
    """Pre-execution guardrail node checking for prompt injection, jailbreaks, and adversarial input."""
    query = state.get("input_query", "")

    with trace_span("guardrail.input_scan", {"query": query[:80]}):
        verdict: GuardrailVerdict = input_guardrail_runnable.invoke(query)
        if not verdict.passed:
            logger.warning(f"Guardrail intercepted input: {verdict.reason}")
            state["is_blocked"] = True
            state["guardrail_reason"] = verdict.reason
            state["current_intent"] = "Blocked"
            state["decision"] = ["Blocked"]
            state["routing_decisions"] = ["Security Guardrail"]
            state["agent_output"] = {
                "Security Guardrail": {
                    "response": verdict.safe_response or "⚠️ Your request was declined by safety guardrails."
                }
            }
        else:
            state["is_blocked"] = False

    return state


# 2. Specialist Node implementations
def conversational_node(state: GraphState) -> GraphState:
    query = state.get("input_query", "")
    model_name = state.get("model_name")
    agent = ConversationalAgent(model_name=model_name)
    result = agent.run(input_query=query)

    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"][agent.name] = result
    return state


def market_analysis_node(state: GraphState) -> GraphState:
    query = state.get("input_query", "")
    model_name = state.get("model_name")
    agent = MarketAnalysisAgent(model_name=model_name)
    result = agent.run(input_query=query)

    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"][agent.name] = result

    if result.get("citations"):
        if "citations" not in state or state["citations"] is None:
            state["citations"] = []
        state["citations"].extend(result["citations"])

    return state


def stock_analyst_node(state: GraphState) -> GraphState:
    query = state.get("input_query", "")
    model_name = state.get("model_name")
    agent = StockAnalyzerAgent(model_name=model_name)
    result = agent.run(input_query=query)

    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"][agent.name] = result
    return state


def finance_qa_node(state: GraphState) -> GraphState:
    query = state.get("input_query", "")
    model_name = state.get("model_name")
    agent = FinanceQAAgent(model_name=model_name)
    result = agent.run(input_query=query)

    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"][agent.name] = result

    if result.get("citations"):
        state["citations"] = result["citations"]
    return state


def stub_node(state: GraphState) -> GraphState:
    intent = state.get("current_intent", "Specialist Agent")
    feature = intent.replace(" Agent", "")
    if "agent_output" not in state or state["agent_output"] is None:
        state["agent_output"] = {}
    state["agent_output"][intent] = {
        "response": f"The **{feature}** capability is currently under active development. In the meantime, please feel free to ask for Market Analysis, Stock Technicals, or Financial Q&A!"
    }
    return state


# 3. Compliance Output Node (LangChain LCEL Runnable)
def compliance_node(state: GraphState) -> GraphState:
    """Post-processes specialist outputs via LangChain output compliance runnable."""
    with trace_span("guardrail.output_compliance"):
        return compliance_runnable.invoke(state)


# Routing conditions
def route_after_guardrail(state: GraphState):
    """Halts execution if input was blocked by security guardrails; otherwise proceeds to triage."""
    if state.get("is_blocked"):
        return END
    return "triage_node"


def route_after_triage(state: GraphState):
    """Directs safe input to the targeted specialist agent node."""
    if state.get("is_blocked"):
        return END

    intent = state.get("current_intent")
    mapping = {
        "Conversational Agent": "conversational_node",
        "Market Analysis Agent": "market_analysis_node",
        "Stock Analyzer Agent": "stock_analyst_node",
        "Finance Q&A Agent": "finance_qa_node",
    }
    return mapping.get(intent, "stub_node")


# Build LangGraph StateGraph
workflow = StateGraph(state_schema=GraphState)

# Register nodes
workflow.add_node("guardrail_node", input_guardrail_node)
workflow.add_node("triage_node", triage_and_route)
workflow.add_node("conversational_node", conversational_node)
workflow.add_node("market_analysis_node", market_analysis_node)
workflow.add_node("stock_analyst_node", stock_analyst_node)
workflow.add_node("finance_qa_node", finance_qa_node)
workflow.add_node("stub_node", stub_node)
workflow.add_node("compliance_node", compliance_node)

# Flow: START -> guardrail_node -> [safe ? triage_node : END]
workflow.add_edge(START, "guardrail_node")
workflow.add_conditional_edges("guardrail_node", route_after_guardrail)

# Flow: triage_node -> [specialist nodes]
workflow.add_conditional_edges("triage_node", route_after_triage)

# Flow: [specialist nodes] -> compliance_node -> END
workflow.add_edge("conversational_node", "compliance_node")
workflow.add_edge("market_analysis_node", "compliance_node")
workflow.add_edge("stock_analyst_node", "compliance_node")
workflow.add_edge("finance_qa_node", "compliance_node")
workflow.add_edge("stub_node", "compliance_node")

workflow.add_edge("compliance_node", END)

orchestrator_graph = workflow.compile(name="Financial AI Orchestrator")
