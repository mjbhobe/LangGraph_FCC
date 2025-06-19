"""
4_router.py - build a basic router, which routes to a tool call or returns
    a message from an LLM

@author: Manish Bhobe
My experiments with AI/ML and Gen AI
Code shared for learning purposes only!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()
console = Console()


# define a tools function
def multiply(a: int, b: int) -> int:
    """function to multiply 2 integers and return the product
    Args:
        a - first integer
        b - second integer
    Returns
        the product a * b
    """
    return a * b


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    max_tokens=1024 * 2,
    timeout=30,
    max_retries=2,
)

# bind the model with tools
model_with_tools = model.bind_tools([multiply])


# tool call node
def llm_inherent_call(message_state: MessagesState) -> MessagesState:
    """node to call the LLM with a tool call"""
    return {"messages": [model_with_tools.invoke(message_state["messages"])]}


# build our graph
builder = StateGraph(MessagesState)
builder.add_node("llm_inherent_call", llm_inherent_call)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "llm_inherent_call")
builder.add_conditional_edges("llm_inherent_call", tools_condition)
builder.add_edge("tools", END)
graph = builder.compile()

console = Console()

while True:
    console.print("[green] Query: [/green]", end="")
    query = input().strip().lower()
    if query in ["exit", "quit"]:
        break
    messages = [HumanMessage(content=query)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
