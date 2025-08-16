"""
4_router.py - build a basic router, which routes to a tool call or returns
    a message from an LLM

@author: Manish Bhobé
My experiments with AI/ML and Gen AI
Code shared for learning purposes only!
"""

from dotenv import load_dotenv
from typing import Union
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()
console = Console()

# define type hint for data type
Number = Union[int, float]


# define my tool functions
def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """function to add (or sum) 2 numbers (integer or float) and return the result
    Args:
        a - first integer (or float)
        b - second integer (or float)
    Returns
        the sum a + b
    """
    return a + b


def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """function to subtract (or take difference of) 2 numbers (integer or float) and return the result
    Args:
        a - first integer (or float)
        b - second integer (or float)
    Returns
        the diference a - b
    """
    return a - b


def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """function to multiply 2 numbers (integer or float) and return the result
    Args:
        a - first integer (or float)
        b - second integer (or float)
    Returns
        the product a * b
    """
    return a * b


def divide(n: Union[int, float], d: Union[int, float]) -> Union[int, float]:
    """function to divide two numbers and return the result
    Args:
        n - the numerator, or first number
        d - the denominator, or second number
    Returns
        the result of n / d
    """
    return n / d if d != 0 else float("inf")


# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.0,
#     max_tokens=1024 * 2,
#     timeout=30,
#     max_retries=2,
# )

model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=1024 * 2,
    timeout=30,
    max_retries=2,
)

# bind the model with tools
tools_list = [add, subtract, multiply, divide]
model_with_tools = model.bind_tools(tools_list)

sys_message = SystemMessage(
    content="You are a helpful assistant taskes with performing math on a set of inputs"
)


# tool call node
def assistant(message_state: MessagesState) -> MessagesState:
    """node to call the LLM with a tool call"""
    return {
        "messages": [model_with_tools.invoke(sys_message + message_state["messages"])]
    }


# build our graph
builder = StateGraph(MessagesState)
# add nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools_list))
builder.add_edge(START, "assistant")
# build structure
builder.add_conditional_edges("assistant", tools_condition)
# loop back from tools to assistant
builder.add_edge("tools", "assistant")
builder.add_edge("tools", END)
graph = builder.compile()

console = Console()

# the biggest problem with this graph is that it will handle
# only a single multiplication (e.g. What is 7 * 8?), but it
# cannot cascade multiple multiplications (e.g. What is 7 * 8 * 9?).

while True:
    console.print("[green] Query: [/green]", end="")
    query = input().strip().lower()
    if query in ["exit", "quit", "bye"]:
        console.print("[red]Exiting...[/red]")
        break
    messages = [HumanMessage(content=query)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
