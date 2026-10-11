from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.specialists.base import BaseSpecialistAgent
from agents.tools.market_tools import (
    fetch_realtime_market_data,
    document_lookup,
    spawn_stock_analyzer,
    reset_agent_citations,
    get_agent_citations
)
from core.config import settings
from observability.logger import get_custom_logger, current_agent_name
from observability.metrics import metrics
from observability.tracer import trace_span
from observability.callbacks import AgentObservabilityCallbackHandler
from utils.helpers import extract_text_from_content

logger = get_custom_logger(name=__name__)

MARKET_ANALYSIS_SYSTEM_PROMPT = """You are an elite, institutional-grade Senior Market Analysis Lead Agent specialized in financial markets, macroeconomic trends, equity research, sector rotation, and market intelligence.

Your goal is to provide comprehensive, well-structured, objective, and deeply insightful market analysis.

You have access to three specialized capabilities:
1. `fetch_realtime_market_data`: Use this to fetch live/recent stock prices, market index summaries (S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX), broad market overviews, historical trends, key financial ratios (P/E, Market Cap), and market news headlines.
2. `document_lookup`: Use this to search internal proprietary market analysis reports, equity research PDFs, and industry insights stored in the MarketAnalysis vector collection.
3. `spawn_stock_analyzer`: Use this to spawn your dedicated **Stock Analyzer Sub-Agent**. The sub-agent is a specialized quantitative technical analyst equipped with real-time and historical technical indicators (RSI, MACD, Bollinger Bands, Moving Averages 20/50/200, Golden/Death Crosses, Stochastic Oscillator, ATR, On-Balance Volume OBV, Volume Surges, Floor Pivot Points, and multi-indicator technical consensus ratings).

### Intelligent Routing & Delegation Guidelines:
- **Individual Stock Technicals & Indicators**:
  - Whenever a query asks about a specific stock's technical setup, whether a stock is overbought or oversold, moving averages, chart levels, RSI, MACD, breakout/squeeze, momentum, support/resistance, or detailed stock evaluation, you MUST spawn the `StockAnalyzerAgent` sub-agent using `spawn_stock_analyzer`.
- **Broad Market & Macro Trends**:
  - For questions on market indices (S&P 500, Nasdaq, VIX, Dow), interest rate impacts, macroeconomic sentiment, or market overview, use `fetch_realtime_market_data`.
- **Internal / Proprietary Research**:
  - For questions referencing analyst reports, equity research PDFs, or internal notes, use `document_lookup`.
- **Multi-Pillar Synthesis**:
  - When appropriate (e.g. evaluating a stock within the context of the broader market or sector), invoke both your sub-agent for quantitative technicals and your market tools for macro/news/document context.

### Final Synthesis Format:
Structure your final answer clearly and professionally:
- **Executive Summary**: Clear, top-line overview and market stance.
- **Quantitative Technical & Indicator Insights**: Highlight key findings from the Stock Analyzer sub-agent (selected indicators, exact values, technical consensus, momentum/trend condition).
- **Macro, Fundamental & Contextual Backing**: Synthesize broader market indices, sector dynamics, or proprietary document insights.
- **Actionable Key Levels**: Support, resistance, and critical breakout thresholds.
- **Strategic Analyst Takeaway**: Catalysts, risks, and balanced outlook (informational purposes only).
"""

class MarketAnalysisAgent(BaseSpecialistAgent):
    """Institutional Market Analysis Lead Agent orchestrating macro research and technical sub-agents."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.tools = [fetch_realtime_market_data, document_lookup, spawn_stock_analyzer]
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=MARKET_ANALYSIS_SYSTEM_PROMPT
        )
        logger.info(f"MarketAnalysisAgent initialized with model: {self.model_name}")

    @property
    def name(self) -> str:
        return "Market Analysis Agent"

    def run(self, input_query: str, **kwargs) -> Dict[str, Any]:
        token = current_agent_name.set(self.name)
        reset_agent_citations()
        with trace_span("agent.market_analysis", {"query": input_query[:80], "model": self.model_name}):
            try:
                logger.info(f"MarketAnalysisAgent executing query: {input_query[:80]}")
                callback = AgentObservabilityCallbackHandler(agent_name=self.name)

                result = self.agent.invoke(
                    input={"messages": [HumanMessage(content=input_query)]},
                    config={"callbacks": [callback]}
                )

                messages = result.get("messages", [])
                final_content = extract_text_from_content(messages[-1].content) if messages else ""
                citations = get_agent_citations()

                metrics.record_agent_execution(self.name, 1.0, success=True)
                logger.info(f"MarketAnalysisAgent completed. Citations captured: {len(citations)}")

                return {
                    "response": final_content,
                    "citations": citations
                }
            except Exception as e:
                metrics.record_agent_execution(self.name, 1.0, success=False)
                logger.error(f"Error in MarketAnalysisAgent execution: {e}", exc_info=True)
                return {
                    "response": f"An error occurred while performing market analysis: {str(e)}",
                    "citations": []
                }
            finally:
                current_agent_name.reset(token)
