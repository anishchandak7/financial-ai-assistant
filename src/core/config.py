import os
import json
from dotenv import load_dotenv
from core.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

# load_dotenv() returns True if it finds and loads any .env file
if load_dotenv():
    logger.info(msg="env variables loaded successfully!")
else:
    # Changed to warning since system-level environment variables 
    # can still be read even if a local .env file isn't present
    logger.warning(msg="No .env file found. Relying on system environment variables.")

class Config:
    
    @property
    def MARKETS(self) -> str | None:
        """Optional market configurations."""
        return os.getenv("MARKETS")

    @property
    def GEMINI_KEY(self) -> str:
        """Required API key for Gemini. Fails fast if missing."""
        key = os.getenv("GEMINI_KEY")
        if not key:
            raise KeyError("Critical environment variable 'GEMINI_KEY' is missing.")
        return key

    @property
    def LLM_CONFIGS(self) -> list:
        """Required JSON array of LLM configs. Validates JSON formatting."""
        raw_configs = os.getenv("LLM_CONFIGS")
        if not raw_configs:
            raise KeyError("Critical environment variable 'LLM_CONFIGS' is missing.")
        
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
        chromadb_path = os.getenv("CHROMA_DB_PATH")
        if chromadb_path is None:
            raise KeyError("'CHROMA_DB_PATH' is not set in env")
        if not os.path.exists(chromadb_path):
            os.makedirs(chromadb_path, exist_ok=True)
        return chromadb_path
    
    @property
    def COLLECTION_NAMES(self) -> list:
        collection_names = os.getenv("COLLECTION_NAMES")
        if collection_names is None:
            raise KeyError("'COLLECTION_NAMES' is not set in env")
        collection_names_list = [collection.strip() for collection in collection_names.split(",")]
        if not collection_names_list:
            raise ValueError("'COLLECTION_NAMES' is set as empty string. Define atleast one collection name from\n(Finance, Portfolio, MarketAnalysis, Goals, News, Tax)")
        return collection_names_list
        

# Instantiate a singleton instance to use throughout your app
settings = Config()