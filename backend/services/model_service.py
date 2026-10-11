from typing import List, Dict, Any, Optional
from core.config import settings
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

class ModelService:
    """Manages LLM configurations, registry, and active model state."""

    def __init__(self) -> None:
        self.llm_configs: List[Dict[str, Any]] = settings.LLM_CONFIGS
        self._active_model_name: str = self.llm_configs[0].get("model_name", "gemini-3.8-flash")
        self._active_display_name: str = self.llm_configs[0].get("display_name", self._active_model_name)

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    @property
    def active_display_name(self) -> str:
        return self._active_display_name

    def set_active_model(self, model_identifier: str) -> str:
        """Sets active model by display_name or model_name."""
        for cfg in self.llm_configs:
            if cfg.get("display_name") == model_identifier or cfg.get("model_name") == model_identifier:
                self._active_model_name = cfg.get("model_name")
                self._active_display_name = cfg.get("display_name")
                logger.info(f"Active model switched to: {self._active_display_name} ({self._active_model_name})")
                return self._active_display_name

        logger.warning(f"Model '{model_identifier}' not found in LLM_CONFIGS. Keeping {self._active_display_name}")
        return self._active_display_name

    def get_available_models(self, details: str = "all") -> Any:
        try:
            if details == "model_name":
                return [m.get("model_name") for m in self.llm_configs]
            if details == "display_name":
                return [m.get("display_name") for m in self.llm_configs]
            return self.llm_configs
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}", exc_info=True)
            raise e

model_service = ModelService()
