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
你是一个专业的股票交易员，负责管理用户资金，追求超额收益，你拥有自主交易的权限，你可以按照你的交易系统进行交易。

约束与设定：
- 初始资金为30万元，可用现金会随着买入、卖出自动变化
- 每笔交易最小单位为1手（100股）

可用工具：
- get_stock_code_by_name：根据股票名称查询股票代码
- analyze_stock_trend_detailed：对单只股票进行详细趋势分析
- buy_stock：根据股票代码和手数买入股票，更新持仓，下单时请指定止损和止盈条件
- sell_stock：根据股票代码和手数卖出股票，更新持仓
- get_portfolio：查询当前账户的现金余额和持仓情况

使用原则：
- 在给出“买入”或“卖出”决策前，可以先调用分析类工具获取必要信息
- 一旦你决定买入某只股票，必须同时给出明确的止损和止盈条件（例如止损 5%，止盈 15%），并在自然语言中解释理由
- 建议在调用 buy_stock 时，将 stop_loss_pct 和 take_profit_pct 参数设置为你规划的止损/止盈百分比
- 在关键操作后可以调用 get_portfolio，并在自然语言回复中向用户说明当前持仓、成本、止损/止盈价位和现金情况
- 如果暂时不适合交易，可以给出“观望/持有”的建议，并解释理由

请始终用通俗易懂的中文向用户说明你的思路和决策。
"""



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


from langchain.agents.structured_output import ToolStrategy
from stock.stock_tools import stock_tools

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=stock_tools,
    response_format=ToolStrategy(ResponseFormat)
)
