from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict, total=False):
    """Execution state passed through the LangGraph multi-agent orchestrator."""
    session_id: Optional[str]  # Session identifier for per-agent memory scoping
    model_name: str
    input_query: str
    current_intent: str
    is_blocked: bool
    guardrail_reason: Optional[str]
    decision: List[str]  # Queue of agents for execution
    routing_decisions: List[str]  # Persistent list of executed agents for UI display
    agent_output: Dict[str, Any]  # Output per agent: {"response": str, "citations": list}
    citations: Optional[List[Dict[str, Any]]]  # Aggregated citations for easy frontend access
    retrieved_contexts: Optional[List[str]]  # Retrieved context chunks for evaluations & feedback
    last_active_agent: Optional[str]  # Agent from previous turn in this session
