from langgraph.graph import StateGraph, START, END
from workflow.state import RouterState
from workflow.nodes import (
    llm_based_router,
    finance_qa_node,
    portfolio_analysis_node,
    market_analysis_node,
    goal_planning_node,
    news_synthesizer_node,
    tax_education_node
)

# 1. Map the LLM output literals to the actual function node names
AGENT_NODE_MAP = {
    "Finance Q&A Agent": "finance_qa_node",
    "Portfolio Analysis Agent": "portfolio_analysis_node",
    "Market Analysis Agent": "market_analysis_node",
    "Goal Planning Agent": "goal_planning_node",
    "News Synthesizer Agent": "news_synthesizer_node",
    "Tax Education Agent": "tax_education_node"
}

# 2. Define the dynamic routing logic
def route_to_next(state: RouterState):
    """Reads the first item in the decision list to decide where to go next."""
    decisions = state.get("decision", [])
    
    # If the list is empty, we are done
    if not decisions:
        return END
        
    # Get the name of the next agent and map it to the node name
    agent = decisions[0]
    return AGENT_NODE_MAP.get(agent, END)

def pop_node(state: RouterState):
    """A utility node that removes the completed agent from the sequence list."""
    decisions = state.get("decision", [])
    # Update the state by returning the list without the first element
    return {"decision": decisions[1:] if decisions else []}


# 3. Build the router workflow
router_workflow = StateGraph(state_schema=RouterState)

# Add the primary router and the utility pop node
router_workflow.add_node("workflow_router", llm_based_router)
router_workflow.add_node("pop_node", pop_node)

# Add all agent nodes
router_workflow.add_node("finance_qa_node", finance_qa_node)
router_workflow.add_node("portfolio_analysis_node", portfolio_analysis_node)
router_workflow.add_node("market_analysis_node", market_analysis_node)
router_workflow.add_node("goal_planning_node", goal_planning_node)
router_workflow.add_node("news_synthesizer_node", news_synthesizer_node)
router_workflow.add_node("tax_education_node", tax_education_node)

# 4. Define the edges (The Flow)
router_workflow.add_edge(START, "workflow_router")

# After LLM router completes, conditionally go to the first agent
router_workflow.add_conditional_edges("workflow_router", route_to_next)

# After an agent finishes executing, it MUST go to the pop_node to remove itself from the list
for node_name in AGENT_NODE_MAP.values():
    router_workflow.add_edge(node_name, "pop_node")

# After popping the completed agent, check the list again and route to the next agent (or END)
router_workflow.add_conditional_edges("pop_node", route_to_next)