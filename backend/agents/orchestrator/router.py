from typing import Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from agents.orchestrator.state import GraphState
from agents.guardrails.prompt_guard import prompt_guard
from core.config import settings
from observability.logger import get_custom_logger
from observability.tracer import trace_span

logger = get_custom_logger(name=__name__)

class AdaptiveRouterResponse(BaseModel):
    intent: Literal[
        "Conversational Agent",
        "Market Analysis Agent",
        "Stock Analyzer Agent",
        "Finance Q&A Agent",
        "Portfolio Analysis Agent",
        "Goal Planning Agent",
        "News Synthesizer Agent",
        "Tax Education Agent"
    ] = Field(
        ...,
        description="The most suitable agent to handle the user query based on intent."
    )
    ticker_symbol: Optional[str] = Field(
        None,
        description="Extracted stock ticker symbol (e.g. 'NVDA', 'AAPL', 'TSLA') if present."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief justification for the routing decision."
    )

ROUTER_SYSTEM_PROMPT = """You are an intelligent triage dispatcher for an institutional Financial AI Assistant.
Analyze the user's input query and route it to the single most appropriate specialist agent:

1. **Conversational Agent**:
   - General greetings ("Hello", "Good morning", "Hi there"), inquiries about capabilities ("What can you do?", "How do you work?"), or general non-financial conversation/questions.
2. **Stock Analyzer Agent**:
   - Queries specifically requesting technical indicator analysis, chart setups, RSI, MACD, Bollinger Bands, Moving Averages, Golden Crosses, support/resistance levels, or momentum for a specific ticker (e.g. "Is NVDA overbought?", "Check AAPL RSI", "What is TSLA support level?").
3. **Market Analysis Agent**:
   - Broad macroeconomic inquiries, US market index status (S&P 500, Dow Jones, Nasdaq, VIX), sector performance, equity research reports, general company performance, or market outlooks.
4. **Finance Q&A Agent**:
   - General finance concepts, definitions, educational topics, 401(k) vs IRA, tax advantages, accounting ratios, or document retrieval questions.
5. **Portfolio Analysis Agent**:
   - User queries specifically asking to analyze or balance personal investment portfolios.
6. **Goal Planning Agent**:
   - Financial retirement planning, wealth target roadmaps, or goal setting.
7. **News Synthesizer Agent**:
   - Breaking financial news aggregation and market sentiment summaries.
8. **Tax Education Agent**:
   - Specific tax filing concepts, capital gains tax rules, or tax brackets.

If a query is ambiguous or casual chit-chat, ALWAYS choose 'Conversational Agent'.
"""

def triage_and_route(state: GraphState) -> GraphState:
    """Pre-evaluates guardrails and determines the adaptive agent routing."""
    input_query = state.get("input_query", "")
    model_name = state.get("model_name") or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")

    with trace_span("orchestrator.triage", {"query": input_query[:80]}):
        # 1. Evaluate Guardrail (Prompt Injection / Adversarial detection)
        guardrail_verdict = prompt_guard.evaluate(input_query)
        if not guardrail_verdict.passed:
            logger.warning(f"Guardrail intercepted input: {guardrail_verdict.reason}")
            state["is_blocked"] = True
            state["guardrail_reason"] = guardrail_verdict.reason
            state["current_intent"] = "Blocked"
            state["decision"] = ["Blocked"]
            state["routing_decisions"] = ["Security Guardrail"]
            state["agent_output"] = {
                "Security Guardrail": {
                    "response": guardrail_verdict.safe_response or "Your request was declined by security guardrails."
                }
            }
            return state

        # 2. Fast heuristic shortcut for common greetings
        stripped = input_query.strip().lower()
        if stripped in ["hi", "hello", "hey", "help", "who are you", "what can you do", "capabilities"]:
            logger.info("Heuristic triage: routed directly to Conversational Agent")
            state["is_blocked"] = False
            state["current_intent"] = "Conversational Agent"
            state["decision"] = ["Conversational Agent"]
            state["routing_decisions"] = ["Conversational Agent"]
            return state

        # 3. LLM-Based Adaptive Routing
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=settings.GEMINI_KEY
            )
            structured_router = llm.with_structured_output(AdaptiveRouterResponse)
            decision: AdaptiveRouterResponse = structured_router.invoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=input_query)
            ])

            selected_agent = decision.intent
            logger.info(f"Adaptive router selected: [{selected_agent}] (reason: {decision.reasoning})")

            state["is_blocked"] = False
            state["current_intent"] = selected_agent
            state["decision"] = [selected_agent]
            state["routing_decisions"] = [selected_agent]
            return state

        except Exception as e:
            logger.error(f"Error during adaptive routing: {e}", exc_info=True)
            # Safe fallback: route to Market Analysis Agent
            fallback = "Market Analysis Agent"
            state["is_blocked"] = False
            state["current_intent"] = fallback
            state["decision"] = [fallback]
            state["routing_decisions"] = [fallback]
            return state
