import akshare as ak
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from langchain.tools import tool
from dataclasses import dataclass


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


@tool
def get_stock_code_by_name(stock_name: str):
    """
    根据股票名称查询股票代码
    
    参数:
        stock_name: 股票名称，如"招商银行"、"贵州茅台"等
    
    返回:
        包含股票代码、名称等信息的字典
    """

    print(f"查询股票名称: {stock_name}", flush=True)
    try:
        # 获取所有A股实时数据
        realtime_df = ak.stock_zh_a_spot_em()
        
        # 模糊匹配股票名称
        matched_stocks = realtime_df[realtime_df['名称'].str.contains(stock_name, na=False)]
        
        if len(matched_stocks) == 0:
            return {"error": f"未找到股票名称包含'{stock_name}'的股票"}
        
        # 返回匹配结果
        results = []
        for _, row in matched_stocks.iterrows():
            results.append({
                "stock_code": row['代码'],
                "stock_name": row['名称'],
                "current_price": row['最新价'],
                "change_pct": row['涨跌幅']
            })
        
        return {
            "query_name": stock_name,
            "matched_count": len(results),
            "stocks": results
        }
        
    except Exception as e:
        return {"error": f"查询股票代码失败: {str(e)}"}


@tool
def get_valid_stock_data(stock_codes):
    """获取有效的股票数据"""
    result = {
        "update_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "stocks": {}
    }
    
    try:
        realtime_df = ak.stock_zh_a_spot_em()
        filtered_df = realtime_df[~realtime_df['代码'].str.startswith('688')]
        filtered_df = filtered_df[~filtered_df['名称'].str.contains('退')]
        filtered_df = filtered_df[~filtered_df['名称'].str.contains('ST')]
        # 过滤掉市值小于100亿的股票
        filtered_df = filtered_df[filtered_df['总市值'] >= 10000*10000*100]


        if stock_codes is None:
            stock_codes = filtered_df['代码'].tolist()
        for code in stock_codes:
            stock_data = filtered_df[filtered_df['代码'] == code]
            
            if len(stock_data) == 0:
                continue
                
            row = stock_data.iloc[0]
            current_price = row['最新价']
            
            # 检查数据是否有效
            if pd.isna(current_price) or current_price == 0:
                continue
            stock_dict = row.to_dict()
            for _, value in stock_dict.items():
                if pd.isna(value):
                    stock_dict = None
            if stock_dict:
                result["stocks"][code] = stock_dict

    except Exception as e:
        result["error"] = str(e)
    
    # 保存到文件
    with open("stock_data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result

@tool
def analyze_stock_trend_detailed(stock_identifier: str, period="30d"):
    """
    详细分析股票趋势，支持股票代码或股票名称
    
    参数:
        stock_identifier: 股票代码（如"000001"）或股票名称（如"平安银行"）
        period: 分析周期，可选值: "7d", "30d", "90d", "180d", "1y"，默认"30d"
    
    返回:
        包含分析周期内的收盘价和成交量异动比的字典
    """
    print(f"分析股票趋势: {stock_identifier}, 周期: {period}", flush=True)
    period_map = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "180d": 180,
        "1y": 365
    }
    
    days = period_map.get(period, 30)
    
    try:
        # 判断输入是股票代码还是股票名称
        stock_code = stock_identifier
        stock_name = None
        
        # 如果输入包含中文，认为是股票名称
        if any('\u4e00' <= char <= '\u9fff' for char in stock_identifier):
            # 先查询股票代码
            realtime_df = ak.stock_zh_a_spot_em()
            matched = realtime_df[realtime_df['名称'].str.contains(stock_identifier, na=False)]
            
            if len(matched) == 0:
                return {"error": f"未找到股票名称包含'{stock_identifier}'的股票"}
            
            if len(matched) > 1:
                # 如果有多个匹配，尝试精确匹配
                exact_match = matched[matched['名称'] == stock_identifier]
                if len(exact_match) > 0:
                    matched = exact_match
                else:
                    # 返回所有匹配的股票供用户选择
                    matches_info = [f"{row['名称']}({row['代码']})" for _, row in matched.iterrows()]
                    return {
                        "error": f"找到多个匹配的股票，请指定具体的股票名称或使用股票代码",
                        "matched_stocks": matches_info
                    }
            
            stock_code = matched.iloc[0]['代码']
            stock_name = matched.iloc[0]['名称']
        
        # 1. 获取数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code, period="daily", 
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
            end_date=datetime.now().strftime('%Y%m%d'),
            adjust="qfq"
        )
        
        # 计算核心指标
        df['MA5'] = df['收盘'].rolling(window=5).mean()
        df['MA20'] = df['收盘'].rolling(window=20).mean()
        df['VOL_MA5'] = df['成交量'].rolling(window=5).mean()
    
        # 截取用户需要的分析周期
        analysis_df = df.tail(days).copy()
        
        # 2. 构造提供给 AI 的“原始感”数据
        recent_data = analysis_df.apply(lambda x: {
            "date": x['日期'],
            "close": round(x['收盘'], 2),
            "vol_change": f"{round((x['成交量']/x['VOL_MA5'] - 1) * 100, 1)}%" # 成交量对比均量
        }, axis=1).tolist()

            
        # 4. 构建返回结构
        return {
            "metadata": {
                "stock_name": stock_name or stock_code,
                "stock_code": stock_code,
                "current_price": round(last_row['收盘'], 2)
            },
            "raw_sequence": {
                "recent_data": recent_data,
                "description": f"这是最近{days}个交易日的收盘价与成交量异动比。"
            }
        }
    except Exception as e:
        return {"error": str(e)}
        
        
    

# ====== 虚拟交易与持仓管理 ======
INITIAL_CASH = 300000.0

PORTFOLIO_FILE = "data/portfolio_state.json"

# 全局虚拟账户状态（单用户场景）

def _load_portfolio_state():
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid portfolio data")
        cash = float(data.get("cash", INITIAL_CASH))
        positions = data.get("positions", {}) or {}
        return {
            'cash': cash,
            'positions': positions,
        }
    except Exception:
        return {
            'cash': INITIAL_CASH,
            'positions': {},
        }


def _save_portfolio_state():
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio_state, f, ensure_ascii=False, indent=4)
    except Exception:
        # 持久化失败时不影响交易逻辑
        pass


portfolio_state = _load_portfolio_state()


def _get_latest_price(stock_code: str):
    '''获取单只股票的最新价格，失败时返回None'''
    try:
        realtime_df = ak.stock_zh_a_spot_em()
        row = realtime_df[realtime_df['代码'] == stock_code]
        if len(row) == 0:
            return None
        price = row.iloc[0]['最新价']
        if pd.isna(price) or price == 0:
            return None
        return float(price)
    except Exception:
        return None


@tool
def buy_stock(stock_code: str, stock_name: str, hands: int, stop_loss_pct: float | None = None, take_profit_pct: float | None = None):
    '''
    虚拟买入股票（不连接真实券商），并为本次交易设定止损/止盈条件（可选）

    参数:
        stock_code: 股票代码，例如'600001'
        stock_name: 股票名称，例如'平安银行'
        hands: 买入手数，1手 = 100股
        stop_loss_pct: 止损百分比，例如 5 表示跌破买入价的 5% 即触发止损
        take_profit_pct: 止盈百分比，例如 15 表示涨幅达到 15% 即触发止盈

    返回:
        包含成交价格、数量、剩余现金和当前持仓及止损/止盈设置的字典
    '''
    global portfolio_state

    if hands <= 0:
        return {'error': '买入手数必须大于0'}

    price = _get_latest_price(stock_code)
    if price is None:
        return {'error': f'无法获取股票 {stock_code} 的最新价格'}

    shares = hands * 100
    cost = price * shares

    if cost > portfolio_state['cash']:
        return {
            'error': '可用现金不足，无法完成买入',
            'cash': round(portfolio_state['cash'], 2),
            'required': round(cost, 2),
        }

    # 更新现金
    portfolio_state['cash'] -= cost

    # 更新持仓
    position = portfolio_state['positions'].get(
        stock_code,
        {
            'name': '',
            'shares': 0,
            'avg_cost': 0.0,
            'stop_loss_pct': None,
            'take_profit_pct': None,
        },
    )
    total_shares = position['shares'] + shares
    if total_shares > 0:
        new_avg_cost = (
            position['avg_cost'] * position['shares'] + cost
        ) / total_shares
    else:
        new_avg_cost = price

    position['name'] = stock_name
    position['shares'] = total_shares
    position['avg_cost'] = new_avg_cost
    # 若本次传入止损/止盈参数，则更新持仓中的设置
    if stop_loss_pct is not None:
        position['stop_loss_pct'] = stop_loss_pct
    if take_profit_pct is not None:
        position['take_profit_pct'] = take_profit_pct
    portfolio_state['positions'][stock_code] = position
    _save_portfolio_state()

    sl_pct = position.get('stop_loss_pct')
    tp_pct = position.get('take_profit_pct')
    stop_loss_price = None
    take_profit_price = None
    if sl_pct is not None:
        stop_loss_price = round(position['avg_cost'] * (1 - sl_pct / 100), 2)
    if tp_pct is not None:
        take_profit_price = round(position['avg_cost'] * (1 + tp_pct / 100), 2)

    return {
        'action': 'buy',
        'stock_code': stock_code,
        'price': round(price, 2),
        'hands': hands,
        'shares': shares,
        'cost': round(cost, 2),
        'cash_after': round(portfolio_state['cash'], 2),
        'position': {
            'shares': position['shares'],
            'avg_cost': round(position['avg_cost'], 2),
            'stop_loss_pct': sl_pct,
            'take_profit_pct': tp_pct,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
        },
    }


@tool
def sell_stock(stock_code: str, hands: int):
    '''
    虚拟卖出股票（不连接真实券商）

    参数:
        stock_code: 股票代码，例如'600519'
        hands: 卖出手数，1手 = 100股

    返回:
        包含成交价格、数量、剩余现金和本次盈亏的字典
    '''
    global portfolio_state

    if hands <= 0:
        return {'error': '卖出手数必须大于0'}

    position = portfolio_state['positions'].get(stock_code)
    if not position or position['shares'] <= 0:
        return {'error': f'当前没有持有股票 {stock_code}，无法卖出'}

    shares = hands * 100
    if shares > position['shares']:
        return {
            'error': '卖出数量超过当前持仓',
            'holding_shares': position['shares'],
            'requested_shares': shares,
        }

    price = _get_latest_price(stock_code)
    if price is None:
        return {'error': f'无法获取股票 {stock_code} 的最新价格'}

    proceeds = price * shares
    portfolio_state['cash'] += proceeds

    # 计算本次实现盈亏
    avg_cost = position['avg_cost']
    realized_profit = (price - avg_cost) * shares

    # 更新持仓数量
    position['shares'] -= shares
    if position['shares'] == 0:
        portfolio_state['positions'].pop(stock_code, None)
    else:
        portfolio_state['positions'][stock_code] = position
    _save_portfolio_state()

    return {
        'action': 'sell',
        'stock_code': stock_code,
        'price': round(price, 2),
        'hands': hands,
        'shares': shares,
        'proceeds': round(proceeds, 2),
        'cash_after': round(portfolio_state['cash'], 2),
        'realized_profit': round(realized_profit, 2),
        'remaining_shares': position['shares'],
    }


@tool
def get_portfolio():
    '''
    获取当前虚拟账户持仓和现金情况

    返回:
        包含现金、持仓列表和估算总资产的字典
    '''
    global portfolio_state

    result = {
        'cash': round(portfolio_state['cash'], 2),
        'positions': [],
    }

    # 尝试获取行情估算市值
    try:
        realtime_df = ak.stock_zh_a_spot_em()
    except Exception:
        realtime_df = None

    total_assets = portfolio_state['cash']

    for code, pos in portfolio_state['positions'].items():
        market_price = None
        market_value = None

        if realtime_df is not None:
            row = realtime_df[realtime_df['代码'] == code]
            if len(row) > 0:
                price = row.iloc[0]['最新价']
                if not pd.isna(price) and price != 0:
                    market_price = float(price)
                    market_value = market_price * pos['shares']
                    total_assets += market_value

        result['positions'].append(
            {
                'stock_code': code,
                'shares': pos['shares'],
                'avg_cost': round(pos['avg_cost'], 2),
                'market_price': round(market_price, 2)
                if market_price is not None
                else None,
                'market_value': round(market_value, 2)
                if market_value is not None
                else None,
            }
        )

    result['total_assets_estimated'] = round(total_assets, 2)
    return result


stock_tools = [
    get_stock_code_by_name,
    get_valid_stock_data,
    analyze_stock_trend_detailed,
    buy_stock,
    sell_stock,
    get_portfolio,
]

# 使用示例
if __name__ == "__main__":
    # data = get_valid_stock_data()
    data = analyze_stock_trend_detailed("000426", "30d")
    print(data)