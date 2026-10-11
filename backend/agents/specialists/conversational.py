from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage
from agents.specialists.base import BaseSpecialistAgent
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

CONVERSATIONAL_SYSTEM_PROMPT = """You are the welcoming AI assistant for the Financial AI Assistant application.

Your role:
1. Greet the user politely, warmly, and professionally.
2. If asked about your capabilities, explain that you are an institutional-grade financial assistant equipped with:
   - **Real-Time Market Analysis**: Live quotes, major US index performance (S&P 500, Nasdaq, Dow Jones, VIX), sector trends, and macro sentiment.
   - **Quantitative Technical Analysis**: Deep technical indicator evaluations (RSI, MACD, Bollinger Bands, Moving Averages 20/50/200, Pivot Points, Volume surges, and Consensus ratings) for individual stocks like NVDA, AAPL, TSLA, MSFT.
   - **Financial Q&A & Document Intelligence**: Retrieval-Augmented Generation (RAG) answering finance, tax, and investment principles citing uploaded documents.
3. If the user asks a completely non-financial question (e.g. cooking recipes, trivia, weather, movie recommendations, sports), politely decline and remind them that you specialize exclusively in financial markets, equities, and economic intelligence. Suggest a relevant financial question they could ask instead.
4. Keep answers concise, clear, and engaging with markdown formatting.
"""

class ConversationalAgent(BaseSpecialistAgent):
    """Handles general greetings, capability overviews, and polite out-of-domain deflections."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY
        )

    @property
    def name(self) -> str:
        return "Conversational Agent"

    def run(self, input_query: str, **kwargs) -> Dict[str, Any]:
        try:
            logger.info(f"ConversationalAgent executing: {input_query[:80]}")
            response = self.llm.invoke([
                SystemMessage(content=CONVERSATIONAL_SYSTEM_PROMPT),
                HumanMessage(content=input_query)
            ])
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
            return {
                "response": str(content),
                "citations": []
            }
        except Exception as e:
            logger.error(f"Error in ConversationalAgent: {e}", exc_info=True)
            return {
                "response": "Hello! I am your Financial AI Assistant. How can I help you with stock analysis or market insights today?",
                "citations": []
            }
