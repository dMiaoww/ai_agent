# pip install -qU langchain "langchain[anthropic]"
from stock.agent_config import agent
from stock_tools import Context



config = {"configurable": {"thread_id": "1"}}
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
                
            # 调用代理
            print("⏳ AI思考中...")
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=context
            )
        
                
            print(f"\n🤖 AI: {response['structured_response'].response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    chat_console(agent)


