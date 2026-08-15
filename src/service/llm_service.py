from google import genai
from core.config import settings
from core.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

class LLMService:
    
    def __init__(self) -> None:
        self.llm_config = settings.LLM_CONFIGS
    
    def available_models(self, details:str = "all"):
        try:
            if details == "model_name":
                return [model.get("model_name") for model in self.llm_config]
            if details == "display_name":
                return [model.get("display_name") for model in self.llm_config]
            return self.llm_config
        except Exception as e:
            logger.error(msg="Failed to fetch available models", exc_info=True, stack_info=True)
            raise e
    

    
    
    
    