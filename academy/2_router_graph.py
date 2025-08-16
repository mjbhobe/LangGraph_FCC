"""
2_router_graph.py - build a basic router graph, where a node
   branches out to 2 possible nodes and then to END

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI
Code shared for learning purposed only!
"""

from typing import TypedDict, Literal
import random
from rich.console import Console
from langgraph.graph import StateGraph, START, END

console = Console()


# create the agent state
class AgentState(TypedDict):
    message: str


"""
In this example, we'll build a router graph where node-1 directs
to node-2 or node-3 depending on some condition
"""


# define the nodes
def node_1(state: AgentState) -> AgentState:
    state["message"] = state["message"] + "I am feeling"
    return state


def node_2(state: AgentState) -> AgentState:
    state["message"] = state["message"] + " happy :)"
    return state


def node_3(state: AgentState) -> AgentState:
    state["message"] = state["message"] + " sad :("
    return state


# this is the function that decided if we branch to node-2 or node-3
def router(state: AgentState) -> Literal["node_2", "node_3"]:
    """router edge that decides where to branch to node-2 or node_3 from node_1"""
    # we use simpla random number to decide
    return "node_3" if random.random() < 0.5 else "node_2"


# build the graph
# START -> node_1 -> router -> node_2 or node_3 -> both to END
graph = StateGraph(AgentState)
graph.add_node("node_1", node_1)
graph.add_node("node_2", node_2)
graph.add_node("node_3", node_3)

# add edges
graph.add_edge(START, "node_1")
graph.add_conditional_edges("node_1", router)
graph.add_edge("node_2", END)
graph.add_edge("node_3", END)
graph = graph.compile()

# invoke the graph
if __name__ == "__main__":
    """run the graph with an initial state"""
    initial_state = AgentState(message="My name is John.")
    final_state = graph.invoke(initial_state)
    print(final_state["message"])
