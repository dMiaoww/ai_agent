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

SYSTEM_PROMPT = """
你是一个专业的股票交易员，请保持冷静的头脑，按照你的交易系统进行交易。
你可以使用的操作有：买入、卖出、持有。请给出你的交易决策。
你的初始资金为30万元, 每笔交易最小单位为1手(100股)。
你可以调用函数来获取你需要的信息。"""



from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek-chat",
    temperature=0.1,
    # timeout=10,
    streaming=True
)

from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Agent响应格式定义"""
    # AI的回复内容（必需）
    response: str
    # 交易决策建议（可选）：买入、卖出、持有
    trading_decision: str | None = None
    # 风险提示（可选）
    risk_warning: str | None = None

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

from langchain.agents.structured_output import ToolStrategy
from stock.stock_tools import stock_tools

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=stock_tools,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)
