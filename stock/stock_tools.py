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
        包含趋势分析、技术指标、支撑阻力位等详细信息的字典
    """
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
        
        # 获取更长时间的数据用于分析
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 3)  # 获取3倍数据用于计算
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"
        )
        
        if len(df) < days:
            return {"error": f"数据不足，只有{len(df)}天数据"}
        
        df = df.sort_values('日期').tail(days)
        df = df.reset_index(drop=True)
        
        # 计算价格序列
        prices = pd.to_numeric(df['收盘'], errors='coerce')
        volumes = pd.to_numeric(df['成交量'], errors='coerce')
        
        # 1. 线性回归趋势
        x = np.arange(len(prices))
        y = prices.values
        slope, intercept = np.polyfit(x, y, 1)
        
        # 2. 移动平均线趋势
        ma_short = prices.rolling(window=5).mean()
        ma_mid = prices.rolling(window=20).mean()
        ma_long = prices.rolling(window=60).mean()
        
        # 3. MACD趋势
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # 4. RSI趋势
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 5. 布林带
        rolling_std = prices.rolling(window=20).std()
        upper_band = ma_mid + 2 * rolling_std
        lower_band = ma_mid - 2 * rolling_std
        
        # 趋势判断
        latest_price = float(prices.iloc[-1])
        latest_ma_short = float(ma_short.iloc[-1]) if not pd.isna(ma_short.iloc[-1]) else latest_price
        latest_ma_mid = float(ma_mid.iloc[-1]) if not pd.isna(ma_mid.iloc[-1]) else latest_price
        latest_ma_long = float(ma_long.iloc[-1]) if not pd.isna(ma_long.iloc[-1]) else latest_price
        
        # 计算趋势得分
        trend_score = 0
        
        # 价格在均线上方
        if latest_price > latest_ma_short:
            trend_score += 1
        if latest_price > latest_ma_mid:
            trend_score += 1
        if latest_price > latest_ma_long:
            trend_score += 1
        
        # 均线排列
        if latest_ma_short > latest_ma_mid > latest_ma_long:
            trend_score += 2
        elif latest_ma_short < latest_ma_mid < latest_ma_long:
            trend_score -= 2
        
        # 斜率方向
        if slope > 0:
            trend_score += 1
        else:
            trend_score -= 1
        
        # MACD方向
        latest_macd = float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0
        if latest_macd > 0:
            trend_score += 1
        else:
            trend_score -= 1
        
        # RSI位置
        latest_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if latest_rsi > 50:
            trend_score += 0.5
        else:
            trend_score -= 0.5
        
        # 判断趋势
        if trend_score >= 4:
            trend = "强烈上涨"
        elif trend_score >= 2:
            trend = "温和上涨"
        elif trend_score >= 0:
            trend = "震荡偏强"
        elif trend_score >= -2:
            trend = "震荡偏弱"
        elif trend_score >= -4:
            trend = "温和下跌"
        else:
            trend = "强烈下跌"
        
        # 计算支撑阻力位
        support_level = float(prices.tail(20).min())
        resistance_level = float(prices.tail(20).max())
        
        # 波动性分析
        daily_returns = prices.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 年化波动率
        
        # 构建详细结果
        result = {
            "stock_code": stock_code,
            "stock_name": stock_name if stock_name else stock_code,
            "period": period,
            "days_analyzed": days,
            "trend_analysis": {
                "trend": trend,
                "trend_score": round(trend_score, 2),
                "regression_slope": round(float(slope), 4),
                "price_change_pct": round((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100, 2),
                "direction": "上涨" if slope > 0 else "下跌",
                "strength": "强" if abs(slope) > prices.mean() * 0.001 else "中等" if abs(slope) > prices.mean() * 0.0005 else "弱"
            },
            "technical_indicators": {
                "current_price": round(latest_price, 2),
                "MA5": round(latest_ma_short, 2),
                "MA20": round(latest_ma_mid, 2),
                "MA60": round(latest_ma_long, 2),
                "MACD": round(latest_macd, 4),
                "RSI": round(latest_rsi, 2),
                "RSI_status": "超买" if latest_rsi > 70 else ("超卖" if latest_rsi < 30 else "正常"),
                "bollinger_position": "上轨" if latest_price > float(upper_band.iloc[-1]) else ("下轨" if latest_price < float(lower_band.iloc[-1]) else "中轨")
            },
            "support_resistance": {
                "support": round(support_level, 2),
                "resistance": round(resistance_level, 2),
                "current_vs_support": round((latest_price - support_level) / (resistance_level - support_level) * 100, 1) if resistance_level > support_level else 50
            },
            "risk_metrics": {
                "volatility": round(volatility, 1),
                "max_drawdown": round((prices.max() - prices.min()) / prices.max() * 100, 1),
                "sharpe_ratio": round(daily_returns.mean() / daily_returns.std() * np.sqrt(252), 2) if daily_returns.std() > 0 else 0
            },
            "volume_analysis": {
                "avg_volume": int(volumes.mean()),
                "latest_volume": int(volumes.iloc[-1]),
                "volume_trend": "放量" if volumes.iloc[-1] > volumes.mean() * 1.2 else "缩量"
            },
        }
        
        return result
        
    except Exception as e:
        return {"error": f"详细分析失败: {str(e)}"}
    
stock_tools = [get_stock_code_by_name, get_valid_stock_data, analyze_stock_trend_detailed]

# 使用示例
if __name__ == "__main__":
    # data = get_valid_stock_data()
    data = analyze_stock_trend_detailed("000426", "30d")
    print(data)