import sys
from typing import TypedDict
from langgraph.graph import StateGraph
from IPython.display import Image, display


class AgentState(TypedDict):
    message: str


def greeting_node(state: AgentState) -> AgentState:
    """simple note that adds a greeting message to the state."""
    state["message"] = "Hey, " + state["message"] + " how was your day?"
    return state


# build the graph
builder = StateGraph(AgentState)

builder.add_node("greeter", greeting_node)

builder.set_entry_point("greeter")
builder.set_finish_point("greeter")
graph = builder.compile()

# display the graph
# display(Image(app.get_graph().draw_mermaid_png()))
# sys.exit(-1)

# run the graph
if __name__ == "__main__":
    # initial_state = AgentState(message="John")
    final_state = graph.invoke({"message": "John"})
    print(final_state["message"])  # Output: Hey, John how was your day?
