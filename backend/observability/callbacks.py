import time
from typing import Dict, Any, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from observability.logger import get_custom_logger, current_agent_name
from observability.metrics import metrics

logger = get_custom_logger(name=__name__)

class AgentObservabilityCallbackHandler(BaseCallbackHandler):
    """LangChain / LangGraph callback handler tracking spans, tokens, and tool latencies."""

    def __init__(self, agent_name: Optional[str] = None):
        super().__init__()
        self.agent_name = agent_name
        self._llm_starts: Dict[str, float] = {}
        self._tool_starts: Dict[str, float] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id, **kwargs: Any
    ) -> None:
        self._llm_starts[str(run_id)] = time.perf_counter()
        agent = self.agent_name or current_agent_name.get() or "UnknownAgent"
        logger.debug(f"LLM Call Started for agent [{agent}] (run_id: {run_id})")

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs: Any) -> None:
        start_time = self._llm_starts.pop(str(run_id), None)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2) if start_time else 0.0

        # Extract token usage metadata if provided by provider
        prompt_tokens = 0
        completion_tokens = 0
        model_name = "gemini-model"

        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            model_name = response.llm_output.get("model_name", model_name)

        if prompt_tokens > 0 or completion_tokens > 0:
            metrics.record_token_usage(model_name, prompt_tokens, completion_tokens)

        agent = self.agent_name or current_agent_name.get() or "UnknownAgent"
        logger.info(
            f"LLM Call Finished: agent=[{agent}] duration={duration_ms}ms prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        )

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        start_time = self._llm_starts.pop(str(run_id), None)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2) if start_time else 0.0
        agent = self.agent_name or current_agent_name.get() or "UnknownAgent"
        logger.error(f"LLM Call Failed: agent=[{agent}] duration={duration_ms}ms error={error}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, *, run_id, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "tool")
        self._tool_starts[str(run_id)] = time.perf_counter()
        logger.info(f"Tool Execution Started: [{tool_name}] with input: {str(input_str)[:150]}")

    def on_tool_end(self, output: str, *, run_id, **kwargs: Any) -> None:
        start_time = self._tool_starts.pop(str(run_id), None)
        duration_sec = (time.perf_counter() - start_time) if start_time else 0.0
        tool_name = kwargs.get("name", "tool")
        metrics.record_tool_call(tool_name=tool_name, duration_sec=duration_sec, success=True)
        logger.info(f"Tool Execution Completed: [{tool_name}] in {round(duration_sec * 1000, 2)}ms")

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs: Any) -> None:
        start_time = self._tool_starts.pop(str(run_id), None)
        duration_sec = (time.perf_counter() - start_time) if start_time else 0.0
        tool_name = kwargs.get("name", "tool")
        metrics.record_tool_call(tool_name=tool_name, duration_sec=duration_sec, success=False)
        logger.error(f"Tool Execution Failed: [{tool_name}] in {round(duration_sec * 1000, 2)}ms with error={error}")
