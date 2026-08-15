from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agents.tools.market_tools import (
    fetch_realtime_market_data,
    document_lookup,
    reset_agent_citations,
    get_agent_citations
)
from core.config import settings
from core.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

MARKET_ANALYSIS_SYSTEM_PROMPT = """You are an expert Market Analysis Agent specialized in financial markets, equity research, sector analysis, macroeconomic trends, and market intelligence.

Your goal is to provide comprehensive, well-structured, objective, and easy-to-understand market analysis.

You have access to two powerful tools:
1. `fetch_realtime_market_data`: Use this to fetch live/recent stock prices, market index summaries (S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX), historical trends, key financial ratios (P/E, Market Cap), and market news.
2. `document_lookup`: Use this to search internal proprietary market analysis reports, equity research PDFs, and industry insights stored in the MarketAnalysis collection.

### ReAct Thought Process Guidelines:
- **Thought**: Analyze the user's inquiry. Determine if you need real-time market data (e.g. current stock prices, index movements, trend statistics), proprietary research documents from the MarketAnalysis knowledge base, or both.
- **Action**: Call the appropriate tool(s) with clear parameters.
- **Observation**: Examine the returned tool data. If more information is needed (e.g. news or history to complement a quote, or document context to explain a price move), make another tool call.
- **Final Answer**: Synthesize the findings into a clear, professional, well-formatted financial market analysis:
  - **Executive Summary**: Clear, direct overview of the situation.
  - **Market & Financial Data**: Present metrics with bullet points, bold numbers, and percentage deltas.
  - **Document & Contextual Insights**: Incorporate any retrieved research notes or analyst reports.
  - **Analyst Takeaway**: Key risks, catalysts, or outlook.
  - Keep tone professional, objective, and easy to read. Note that insights are for educational/informational purposes only.
"""

class MarketAnalysisAgent:
    
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.6-flash")
        self.tools = [fetch_realtime_market_data, document_lookup]
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY
        )
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=MARKET_ANALYSIS_SYSTEM_PROMPT
        )
        logger.info(msg=f"MarketAnalysisAgent initialized with model: {self.model_name}")

    def run(self, input_query: str) -> Dict[str, Any]:
        """Runs the ReAct market analysis agent on the user query."""
        try:
            logger.info(msg=f"MarketAnalysisAgent executing query: {input_query}")
            
            # Reset citation registry for this execution
            reset_agent_citations()
            
            # Execute ReAct agent loop
            result = self.agent.invoke(
                input={"messages": [HumanMessage(content=input_query)]}
            )
            
            # Extract final message content
            messages = result.get("messages", [])
            final_content = ""
            if messages:
                last_message = messages[-1]
                raw_content = last_message.content
                if isinstance(raw_content, str):
                    final_content = raw_content
                elif isinstance(raw_content, list):
                    text_blocks = []
                    for block in raw_content:
                        if isinstance(block, dict) and "text" in block:
                            text_blocks.append(block["text"])
                        elif isinstance(block, str):
                            text_blocks.append(block)
                    final_content = "\n".join(text_blocks)
                else:
                    final_content = str(raw_content)

            # Retrieve any citations captured from document_lookup
            citations = get_agent_citations()
            logger.info(msg=f"MarketAnalysisAgent completed. Citations captured: {len(citations)}")
            
            return {
                "response": final_content,
                "citations": citations
            }
            
        except Exception as e:
            logger.error(msg=f"Error in MarketAnalysisAgent execution: {e}", exc_info=True)
            return {
                "response": f"An error occurred while performing market analysis: {str(e)}",
                "citations": []
            }
