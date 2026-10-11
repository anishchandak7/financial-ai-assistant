from typing import Dict, Any, Optional
from agents.orchestrator.state import GraphState
from agents.orchestrator.graph import orchestrator_graph
from services.model_service import model_service
from observability.logger import get_custom_logger, current_trace_id, current_request_id
from observability.tracer import trace_span
from observability.langsmith_tracer import get_langsmith_tracer

logger = get_custom_logger(name=__name__)

class OrchestrationService:
    """Coordinates execution of the multi-agent orchestrator graph with LangSmith telemetry."""

    def __init__(self) -> None:
        self.workflow = orchestrator_graph

    def set_model(self, selected_model: str):
        model_service.set_active_model(selected_model)

    def execute(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes the adaptive multi-agent graph with guardrails, per-agent memory, and telemetry."""
        import uuid
        from agents.tools.memory_tools import set_current_session_id

        effective_session_id = session_id or str(uuid.uuid4())
        set_current_session_id(effective_session_id)

        if model_override:
            self.set_model(model_override)

        active_model = model_service.active_model_name
        logger.info(f"Orchestrating user prompt (Session: {effective_session_id[:8]}...) using model: {active_model}")

        state: GraphState = {
            "session_id": effective_session_id,
            "model_name": active_model,
            "input_query": prompt,
            "agent_output": {},
            "citations": []
        }

        # Build LangSmith configuration and run metadata
        run_id = str(uuid.uuid4())
        config: Dict[str, Any] = {
            "run_name": "Financial AI Assistant Orchestration",
            "run_id": run_id,
            "tags": ["financial-assistant", "multi-agent", active_model],
            "metadata": {
                "session_id": effective_session_id,
                "model_name": active_model,
                "trace_id": current_trace_id.get(),
                "request_id": current_request_id.get()
            }
        }

        ls_tracer = get_langsmith_tracer()
        if ls_tracer:
            config["callbacks"] = [ls_tracer]

        with trace_span("orchestrator.execute", {"model": active_model, "query": prompt[:80], "session_id": effective_session_id}):
            try:
                result = self.workflow.invoke(input=state, config=config)
                result["session_id"] = effective_session_id
                logger.info("Multi-agent workflow execution completed successfully.")

                # If LangSmith tracing is active, attach reference-free evaluation feedback asynchronously
                from core.config import settings
                if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
                    retrieved_contexts = result.get("retrieved_contexts")
                    if not retrieved_contexts and isinstance(result.get("agent_output"), dict):
                        for out in result["agent_output"].values():
                            if isinstance(out, dict) and out.get("retrieved_contexts"):
                                retrieved_contexts = out["retrieved_contexts"]
                                break

                    if retrieved_contexts:
                        primary_answer = ""
                        if isinstance(result.get("agent_output"), dict):
                            for out in result["agent_output"].values():
                                if isinstance(out, dict) and out.get("response"):
                                    primary_answer = out["response"]
                                    break

                        if primary_answer:
                            import threading
                            from observability.langsmith_tracer import log_rag_run_feedback

                            def _bg_eval(target_run_id: str, query_text: str, contexts_list: list, response_text: str):
                                try:
                                    from evals.ragas_evaluator import ragas_evaluator
                                    scores = ragas_evaluator.evaluate_single_turn(
                                        query=query_text,
                                        contexts=contexts_list,
                                        response=response_text
                                    )
                                    log_rag_run_feedback(
                                        run_id=target_run_id,
                                        faithfulness=scores.get("faithfulness", 0.0),
                                        answer_relevance=scores.get("answer_relevance", 0.0),
                                        context_relevance=scores.get("context_relevance", 0.0),
                                        overall_rag_score=scores.get("overall_rag_score", 0.0)
                                    )
                                except Exception as eval_err:
                                    logger.warning(f"Background LangSmith evaluation feedback failed for run {target_run_id}: {eval_err}")

                            threading.Thread(
                                target=_bg_eval,
                                args=(run_id, prompt, retrieved_contexts, primary_answer),
                                daemon=True
                            ).start()

                return result
            except Exception as e:
                logger.error(f"Failed to execute multi-agent workflow: {e}", exc_info=True)
                raise e

orchestration_service = OrchestrationService()
