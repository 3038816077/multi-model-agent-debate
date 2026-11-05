from langgraph.graph import StateGraph, END
from app.agent.state import State
from app.agent.nodes import llm_node, tool_node

def route(state: State):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool"
    return END

graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", route)
graph.add_edge("tool", "llm")
agent_app = graph.compile()