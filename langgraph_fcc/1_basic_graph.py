from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# create the agent state
class AgentState(TypedDict):
    message: str


# create the single node for graph
def hello_node(state: AgentState) -> AgentState:
    """modified message in the agent state to add greeting"""
    state["message"] = "Hey there " + state["message"] + "! How was your day?"
    return state


# build the graph
# START -> hello_node -> END
graph = StateGraph(AgentState)
graph.add_node("hello", hello_node)
graph.add_edge(START, "hello")
graph.add_edge("hello", END)
graph = graph.compile()

# invoke the graph
if __name__ == "__main__":
    """run the graph with an initial state"""
    initial_state = AgentState(message="John")
    final_state = graph.invoke(initial_state)
    print(final_state["message"])
