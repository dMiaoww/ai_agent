# pip install -qU langchain "langchain[anthropic]"
from stock.agent_config import agent
from stock.stock_tools import Context



config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
## 封装成界面
def chat_console(agent):
    """控制台聊天界面"""
    print("🤖 AI助手已启动！输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    context = Context(user_id="1")
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
                
            if not user_input:
                continue
                
            # 调用代理（流式输出）
            print("\n🤖 AI: ", end="", flush=True)
            
            response_text = ""
            final_result = None
            
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=context,
                stream_mode="values"
            ):
                # 保存最终结果
                final_result = event
                
                # 尝试从 structured_response 中获取内容
                if "structured_response" in event:
                    structured = event["structured_response"]
                    if hasattr(structured, 'response'):
                        new_text = structured.response
                        if len(new_text) > len(response_text):
                            new_content = new_text[len(response_text):]
                            print(new_content, end="", flush=True)
                            response_text = new_text
                # 如果没有 structured_response，尝试从 messages 中获取
                elif "messages" in event:
                    messages = event["messages"]
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            # 如果是字符串内容
                            if isinstance(last_message.content, str):
                                if len(last_message.content) > len(response_text):
                                    new_content = last_message.content[len(response_text):]
                                    print(new_content, end="", flush=True)
                                    response_text = last_message.content
            
            print()  # 换行
            
            # 如果有交易决策或风险提示，显示出来
            if final_result and "structured_response" in final_result:
                structured = final_result["structured_response"]
                if hasattr(structured, 'trading_decision') and structured.trading_decision:
                    print(f"\n📊 交易建议: {structured.trading_decision}")
                if hasattr(structured, 'risk_warning') and structured.risk_warning:
                    print(f"⚠️  风险提示: {structured.risk_warning}")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    chat_console(agent)


