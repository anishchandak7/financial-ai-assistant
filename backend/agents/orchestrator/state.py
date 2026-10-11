from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    """Execution state passed through the LangGraph multi-agent orchestrator."""
    model_name: str
    input_query: str
    current_intent: str
    is_blocked: bool
    guardrail_reason: Optional[str]
    decision: List[str]  # Queue of agents for execution
    routing_decisions: List[str]  # Persistent list of executed agents for UI display
    agent_output: Dict[str, Any]  # Output per agent: {"response": str, "citations": list}
    citations: Optional[List[Dict[str, Any]]]  # Aggregated citations for easy frontend access
