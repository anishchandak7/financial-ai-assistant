from core.config import settings
from core.logger import get_custom_logger
from workflow.state import RouterState
from workflow.router import router_workflow

logger = get_custom_logger(name=__name__)

class WorkflowService:
    
    def __init__(self) -> None:
        self.model_config = None

    def set_model(self, selected_model:str):
        if self.model_config is not None:
            logger.info(msg=f"Workflow is already configured for {self.model_config.get("display_name")}")
            return
        llm_configs = settings.LLM_CONFIGS
        for model_dict in llm_configs:
            if model_dict.get("display_name") == selected_model:
                self.model_config = model_dict
                break
        logger.info(msg=f"Workflow is set to use {selected_model}")
        return
    
    def execute(self, prompt:str):
        
        try:
            # initialize the state.
            state = RouterState(
                model_name=self.model_config.get("model_name"),
                input_query=prompt
            )
            compiled_workflow = router_workflow.compile(name="Finance AI Assistant Router")
            result = compiled_workflow.invoke(
                input=state
            )
            logger.info(msg=f"Workflow response: {result}")
            return result
        except Exception as e:
            logger.error(msg="Failed to execute the workflow", exc_info=True, stack_info=True)
            raise e
                