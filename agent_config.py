# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from dotenv import load_dotenv
import os


load_dotenv()

# SYSTEM_PROMPT = """You are an expert weather forecaster.

# You have access to two tools:

# - get_weather_for_location: use this to get the weather for a specific location
# - get_user_location: use this to get the user's location

# If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

SYSTEM_PROMPT = """You are an assistant specialized in providing information and controlling a robot."""


from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

from langchain.agents.structured_output import ToolStrategy
from mytools import tools, Context
agent = create_agent(
    model=model,
    system_prompt= SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)
