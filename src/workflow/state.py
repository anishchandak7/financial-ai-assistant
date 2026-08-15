from typing import TypedDict, List, Dict, Any, Optional

class RouterState(TypedDict, total=False):
    model_name: str
    input_query: str
    decision: List[str]  # Active queue of agents popped during execution
    routing_decisions: List[str]  # Persistent list of routed agents for UI display
    agent_output: Dict[str, Any]  # Output per agent: str or {"response": str, "citations": list}
    citations: Optional[List[Dict[str, Any]]]  # Aggregated citations for easy frontend access