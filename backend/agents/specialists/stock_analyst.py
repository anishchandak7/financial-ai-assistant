from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.specialists.base import BaseSpecialistAgent
from agents.tools.stock_analyzer_tools import (
    calculate_technical_indicators,
    get_indicator_recommendations
)
from core.config import settings
from observability.logger import get_custom_logger, current_agent_name
from observability.metrics import metrics
from observability.tracer import trace_span
from observability.callbacks import AgentObservabilityCallbackHandler
from utils.helpers import extract_text_from_content

logger = get_custom_logger(name=__name__)

STOCK_ANALYZER_SYSTEM_PROMPT = """You are an elite, quantitative Stock Analyzer Sub-Agent specialized in technical analysis, price action mechanics, and market microstructure.

Your mission is to provide rigorous, data-driven, objective, and actionable technical analysis for individual stocks using technical indicators calculated from real-time and historical market data.

You have access to two specialized tools:
1. `calculate_technical_indicators(symbol, indicator_category, period)`:
   - Computes verified mathematical indicators from yfinance historical OHLCV data.
   - Categories:
     - 'momentum': RSI (14), MACD (12,26,9), Stochastic Oscillator, Rate of Change (ROC).
     - 'trend': SMA (20, 50, 200), EMA (12, 26, 50), Golden/Death Cross detection, price vs MAs.
     - 'volatility': Bollinger Bands (%B, Bandwidth Squeeze alert), ATR (14), 52-week range position.
     - 'volume': 20-day Volume SMA, Volume Surge ratio, On-Balance Volume (OBV) trend.
     - 'support_resistance': Classical Floor Pivots (P, S1, S2, R1, R2), 20-day Swing High/Low.
     - 'all': Complete multi-pillar profile with overall quantitative consensus rating.
2. `get_indicator_recommendations(symbol, query_intent)`:
   - Suggests the ideal indicator suite tailored to the specific query intent.

### ReAct Reasoning & Execution Guidelines:
1. **Thought - Identify Intent & Select Suitable Indicators**:
   - Analyze the ticker symbol and the specific user query.
   - Determine which indicators are most suitable for the question:
     - Asking about **overbought/oversold, momentum, or reversal**? -> Prioritize **RSI, Stochastic, MACD**.
     - Asking about **trend direction, bull vs bear, or moving averages**? -> Prioritize **SMA (20/50/200), EMA, Crossover signals**.
     - Asking about **volatility, squeeze, or breakout**? -> Prioritize **Bollinger Bands (%B, Squeeze), ATR**.
     - Asking about **volume or institutional accumulation**? -> Prioritize **OBV, Volume Surge ratio**.
     - Asking about **key levels, entry, exit, target, or stop loss**? -> Prioritize **Pivot Points (P, S1, S2, R1, R2) and Swing High/Low**.
     - Asking for a **general technical review or stock outlook**? -> Use **'all'** for the full 4-pillar consensus.
2. **Action**: Call `calculate_technical_indicators` with the appropriate category (or 'all').
3. **Observation**: Scrutinize the mathematical outputs, status signals, and divergences.
4. **Final Answer Formulation**:
   Format the technical analysis into a professional, highly readable report:
   - **Executive Technical Summary**: Ticker, current price, and primary technical posture.
   - **Targeted Indicator Deep-Dive**: Present the chosen indicators with exact numerical values and explain what they mean for this specific query.
   - **Indicator Alignment & Signals**: Highlight confirmations or conflicts (e.g. bullish trend but overbought momentum).
   - **Actionable Price Levels**: Detail exact Support ($S_1, S_2$) and Resistance ($R_1, R_2$) thresholds.
   - **Technical Consensus & Outlook**: Clear directional bias and risk considerations. Note that insights are for educational/informational purposes only.
"""

from memory.manager import agent_memory_manager
from agents.tools.memory_tools import set_current_session_id

class StockAnalyzerAgent(BaseSpecialistAgent):
    """Quantitative Technical Analysis specialist agent calculating mathematical indicators with session memory."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.tools = [calculate_technical_indicators, get_indicator_recommendations]
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=STOCK_ANALYZER_SYSTEM_PROMPT
        )
        logger.info(f"StockAnalyzerAgent initialized with model: {self.model_name}")

    @property
    def name(self) -> str:
        return "Stock Analyzer Agent"

    def run(self, input_query: str, ticker: Optional[str] = None, session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        token = current_agent_name.set(self.name)
        session_token = None
        if session_id:
            session_token = set_current_session_id(session_id)

        with trace_span("agent.stock_analyst", {"ticker": ticker, "query": input_query[:80]}):
            try:
                # If ticker is missing, check if session memory has an active ticker
                effective_ticker = ticker
                if not effective_ticker and session_id:
                    session_info = agent_memory_manager.get_session_summary_for_router(session_id)
                    effective_ticker = session_info.get("active_ticker")

                formatted_query = f"Ticker: {effective_ticker.upper()}\nQuery: {input_query}" if effective_ticker else input_query

                # Check for existing working memory scratchpad
                working_summary = ""
                if session_id:
                    working_summary = agent_memory_manager.get_working_memory_summary(session_id, self.name)
                if working_summary:
                    formatted_query = f"{working_summary}\n\n{formatted_query}"

                logger.info(f"StockAnalyzerAgent executing: {formatted_query[:100]} (Session: {session_id})")

                # Retrieve dialogue history for Stock Analyzer Agent
                agent_messages = []
                if session_id:
                    history = agent_memory_manager.get_messages_for_agent(session_id, self.name, max_turns=6)
                    agent_messages.extend(history)
                agent_messages.append(HumanMessage(content=formatted_query))

                callback = AgentObservabilityCallbackHandler(agent_name=self.name)
                result = self.agent.invoke(
                    input={"messages": agent_messages},
                    config={"callbacks": [callback]}
                )

                messages = result.get("messages", [])
                final_content = extract_text_from_content(messages[-1].content) if messages else ""
                metrics.record_agent_execution(self.name, 1.0, success=True)

                # Persist turn in Stock Analyzer's dedicated memory partition
                if session_id:
                    agent_memory_manager.add_turn(
                        session_id=session_id,
                        agent_name=self.name,
                        user_query=input_query,
                        agent_response=final_content,
                        metadata={"ticker": effective_ticker.upper() if effective_ticker else None}
                    )
                    if effective_ticker:
                        agent_memory_manager.update_working_memory(
                            session_id=session_id,
                            agent_name=self.name,
                            key="last_analyzed_ticker",
                            value=effective_ticker.upper()
                        )

                return {
                    "response": final_content,
                    "ticker": effective_ticker.upper() if effective_ticker else None
                }
            except Exception as e:
                metrics.record_agent_execution(self.name, 1.0, success=False)
                logger.error(f"Error in StockAnalyzerAgent execution: {e}", exc_info=True)
                return {
                    "response": f"Error during stock technical analysis: {str(e)}",
                    "ticker": ticker
                }
            finally:
                current_agent_name.reset(token)
