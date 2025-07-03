"""
5_agent_react.py - configure a basic math agent with react pattern,
    which should be able to handle cascading math ops, such as 4 *7 / 12 + 45

@author: Manish Bhobe
My experiments with AI/ML and Gen AI
Code shared for learning purposes only!
"""

import math, sys
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


# define basic math ops as tools
def add(a: Number, b: Number) -> Number:
    """function to add 2 numbers and return the sum
    Args:
        a - first number
        b - second number
    Returns
        the sum a + b
    """
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """function to subtract 2 numbers and return the difference
    Args:
        a - first number
        b - second number
    Returns
        the difference a - b
    """
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """function to multiply 2 numbers and return the product
    Args:
        a - first number
        b - second number
    Returns
        the product a * b
    """
    return a * b


def divide(n: Number, d: Number) -> Number:
    """function to divide 2 numbers and return the quotient
    Args:
        n - numerator
        d - denominator
    Returns
        the quotient n / d
    """
    return (n / d) if d != 0 else math.nan


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.0,
    max_tokens=1024 * 2,
    timeout=30,
    max_retries=2,
)

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.0,
#     max_tokens=1024 * 2,
#     timeout=30,
#     max_retries=2,
# )

# bind the model with tools
tools_list = [add, subtract, multiply, divide]
model_with_tools = model.bind_tools(tools_list)


# here is my main LLM assistant
system_message = SystemMessage(
    "You are a helpful assistant tasked with solving math problems."
)


def assistant(message_state: MessagesState) -> MessagesState:
    """node to call the LLM with a tool call"""
    return {
        "messages": [
            model_with_tools.invoke([system_message] + message_state["messages"])
        ]
    }


console = Console()

# build our graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools_list))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", END)
graph = builder.compile()
# display as ascii
# console.print(Markdown(graph.get_graph().ascii()), justify="center")
print(graph.get_graph().print_ascii())

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
