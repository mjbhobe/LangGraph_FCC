from operator import add
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# Define state schema
class State(TypedDict):
    count: Annotated[list[int], add]  # Concatenates integers
    log: Annotated[list[AnyMessage], add_messages]  # Concatenates messages


# define our nodes
def node_1(state):
    return {
        "count": [state["count"][-1] + 1],
        "log": [state["log"][-1]] + [HumanMessage("I'm from Node 1")],
    }


def node_2(state):
    return {
        "count": [state["count"][-1] + 1],
        "log": [state["log"][-1]] + [HumanMessage("I'm from Node 2")],
    }


def node_3(state):
    return {
        "count": [state["count"][-1] + 1],
        "log": [state["log"][-1]] + [HumanMessage("I'm from Node 3")],
    }


# build out the graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
# Define edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)
# Compile the graph
graph = builder.compile()

# invoke the graph
# Initial state
initial_state = {"count": [1], "log": [HumanMessage("Start")]}

# Invoke the graph
result = graph.invoke(initial_state)

# Display results
print("================================ Add Operator =================================")
print(result["count"])  # Tracks state updates for `count`

print(
    "\n================================ Add Message ================================="
)
print(result["log"])  # Tracks message state
