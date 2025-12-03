import os
from openai import OpenAI
import json
# from dotenv import load_dotenv

# 加载环境变量
# load_dotenv()

# 初始化 OpenAI 客户端
client = OpenAI(api_key="sk-84adf8bfd04144fda49637565f3f51b5", base_url="https://api.deepseek.com")

# 定义工具函数
def get_current_weather(location, unit="celsius"):
    """模拟获取天气数据"""
    # 在实际应用中，这里会调用真实的天气 API
    weather_data = {
        "location": location,
        "temperature": "22",
        "unit": unit,
        "forecast": ["晴朗", "微风"],
        "humidity": "65%"
    }
    return json.dumps(weather_data, ensure_ascii=False)

def calculate_bmi(weight, height):
    """计算 BMI"""
    bmi = weight / (height ** 2)
    category = "体重过轻" if bmi < 18.5 else "正常范围" if bmi < 24 else "体重过重"
    result = {
        "bmi": round(bmi, 2),
        "category": category,
        "weight": weight,
        "height": height
    }
    return json.dumps(result, ensure_ascii=False)

def move(direction):
    """模拟机器人向前移动"""
    print(f"机器人正在向{direction}移动...")
    return "开始移动机器人：" + direction

# 定义可供调用的工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "让机器人向指定方向移动，方向为字符串",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "移动方向，例如：front、back、turn left、turn right"
                    }
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "根据身高体重计算BMI指数",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight": {
                        "type": "number",
                        "description": "体重，单位：千克",
                    },
                    "height": {
                        "type": "number", 
                        "description": "身高，单位：米",
                    },
                },
                "required": ["weight", "height"],
            },
        }
    }
]

def run_agent_conversation(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    print(f"用户: {user_input}")
    
    try:
        # 第一步：调用模型，让它决定是否要调用函数
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        print(f"AI 初始回复: {response_message.content}")
        
        # 第二步：检查模型是否想要调用函数
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"AI 决定调用函数: {function_name}")
                print(f"函数参数: {function_args}")
                
                # 执行对应的函数
                if function_name == "get_current_weather":
                    function_response = get_current_weather(**function_args)
                elif function_name == "calculate_bmi":
                    function_response = calculate_bmi(**function_args)
                elif function_name == "move":
                    function_response = move(**function_args)
                else:
                    function_response = f"错误：未知函数 {function_name}"
                
                # 将函数执行结果返回给模型
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
            
            # 让模型基于函数结果生成最终回复
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            
            final_message = second_response.choices[0].message.content
            print(f"AI 最终回复: {final_message}")
            return final_message
        
        return response_message.content
    
    except Exception as e:
        return f"发生错误: {str(e)}"

# 测试函数调用
if __name__ == "__main__":
    # 测试用例
    test_cases = [
        "往前一点, 你自己看着办",  # 调用 move 函数
        "往左转",  # 调用 move 函数
        # "简单的问候，不需要调用任何函数"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试用例 {i}:")
        print('='*50)
        result = run_agent_conversation(test_case)
        print(f"\n结果: {result}")