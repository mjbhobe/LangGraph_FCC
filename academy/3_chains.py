from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
console = Console()

# setup a collection of messages
messages = [SystemMessage("You are an expert in marine biology")]
messages.extend(
    [AIMessage("So you said you were researching ocean mammals?", name="Model")]
)
messages.extend([HumanMessage("Yes, that's right", name="Myself")])
messages.extend(
    [AIMessage("Great! What would you like to learn about?", name="Myself")]
)
messages.extend(
    [
        HumanMessage(
            "I want to know the best place to see Orcas in the US", name="Myself"
        )
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    max_tokens=1024 * 2,
    timeout=30,
    max_retries=2,
)

response = model.invoke(messages)
console.print(Markdown(response.content))


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


model_with_tools = model.bind_tools([multiply])

response = model_with_tools.invoke(
    [
        HumanMessage(
            "If I saw a batch of 4 orcas 3 times, how many oracs did I see?",
            name="Myself",
        )
    ]
)
console.print(response)
