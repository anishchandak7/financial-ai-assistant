import os
import json
from pathlib import Path
from dotenv import load_dotenv
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

# Locate backend directory and project workspace root
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DOTENV_PATH = BACKEND_DIR / ".env"

if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH)
    logger.info(msg=f"env variables loaded successfully from {DOTENV_PATH}!")
elif (PROJECT_ROOT / ".env").exists():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    logger.info(msg=f"env variables loaded successfully from {PROJECT_ROOT / '.env'}!")
elif load_dotenv():
    logger.info(msg="env variables loaded successfully!")
else:
    logger.warning(msg="No .env file found. Relying on system environment variables.")

class Config:
    
    @property
    def MARKETS(self) -> str | None:
        """Optional market configurations."""
        return os.getenv("MARKETS", "US, GB, ASIA, EUROPE, RATES, COMMODITIES, CURRENCIES, CRYPTOCURRENCIES")

    @property
    def GEMINI_KEY(self) -> str:
        """Required API key for Gemini. Fails fast if missing."""
        key = os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise KeyError("Critical environment variable 'GEMINI_KEY' (or 'GEMINI_API_KEY') is missing.")
        return key

    @property
    def LLM_CONFIGS(self) -> list:
        """Required JSON array of LLM configs. Validates JSON formatting."""
        raw_configs = os.getenv("LLM_CONFIGS")
        if not raw_configs:
            return [
                {"display_name": "Gemini 3.8 Flash", "model_name": "gemini-3.8-flash"},
                {"display_name": "Gemini 3.6 Flash", "model_name": "gemini-3.6-flash"},
            ]
        
        try:
            parsed_json = json.loads(raw_configs)
            if not isinstance(parsed_json, list):
                raise TypeError("'LLM_CONFIGS' must be a JSON array (list).")
            return parsed_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse 'LLM_CONFIGS' as valid JSON: {e}")

    @property
    def CHROMA_DB_PATH(self) -> str:
        """Path to Chroma DB. Check if folder structure is present or not."""
        chromadb_path = os.getenv("CHROMA_DB_PATH", "vectorstore/chromadb")
        if not os.path.isabs(chromadb_path):
            chromadb_path = str(PROJECT_ROOT / chromadb_path)
        if not os.path.exists(chromadb_path):
            os.makedirs(chromadb_path, exist_ok=True)
        return chromadb_path
    
    @property
    def COLLECTION_NAMES(self) -> list:
        collection_names = os.getenv("COLLECTION_NAMES", "Finance, Portfolio, MarketAnalysis, Goals, News, Tax")
        collection_names_list = [collection.strip() for collection in collection_names.split(",")]
        if not collection_names_list:
            raise ValueError("'COLLECTION_NAMES' is set as empty string. Define atleast one collection name from\n(Finance, Portfolio, MarketAnalysis, Goals, News, Tax)")
        return collection_names_list

    @property
    def LANGSMITH_TRACING(self) -> bool:
        """Checks if LangSmith tracing is enabled AND a valid non-empty API key is present."""
        val = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
        is_true = val.strip().lower() in ("true", "1", "yes") if val else False
        has_key = bool(self.LANGSMITH_API_KEY and self.LANGSMITH_API_KEY.strip())
        return is_true and has_key

    @property
    def LANGSMITH_API_KEY(self) -> str | None:
        """API key for LangSmith tracing."""
        key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        return key.strip() if key and key.strip() else None

    @property
    def LANGSMITH_PROJECT(self) -> str:
        """Target project name in LangSmith dashboard."""
        return os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "financial-ai-assistant")

    @property
    def LANGSMITH_ENDPOINT(self) -> str:
        """LangSmith API endpoint."""
        return os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    def configure_langsmith_environment(self) -> None:
        """Synchronizes LangSmith environment variables for automatic LangChain / LangGraph tracing."""
        if self.LANGSMITH_API_KEY and self.LANGSMITH_TRACING:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
            logger.info(f"LangSmith Tracing ENABLED: Project='{self.LANGSMITH_PROJECT}'")
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"
            logger.info("LangSmith Tracing not active (set LANGSMITH_API_KEY and LANGSMITH_TRACING=true in .env to enable).")

# Instantiate a singleton instance to use throughout your app
settings = Config()
settings.configure_langsmith_environment()