"""
Gradio Web界面 - 股票分析AI助手
支持流式输出和工具调用可视化
"""
import gradio as gr
from stock.agent_config import agent
from stock.stock_tools import Context
import json
from datetime import datetime


config = {"configurable": {"thread_id": "1"}, "recursion_limit": 50}
context = Context(user_id="1")


def format_tool_call(tool_name, tool_input):
    """格式化工具调用信息"""
    return f"""
    <div style="background-color: #f0f8ff; padding: 8px; margin: 3px 0; border-radius: 5px; border-left: 4px solid #4CAF50;">
        <strong>🔧 调用工具:</strong> <code>{tool_name}</code>
    </div>
    """


def format_tool_result(tool_name, success=True):
    """格式化工具返回结果"""
    if success:
        status_icon = "✅"
        status_text = "成功"
        bg_color = "#f0fff0"
        border_color = "#4CAF50"
    else:
        status_icon = "❌"
        status_text = "失败"
        bg_color = "#fff5f5"
        border_color = "#f44336"
    
    return f"""
    <div style="background-color: {bg_color}; padding: 8px; margin: 3px 0; border-radius: 5px; border-left: 4px solid {border_color};">
        <strong>{status_icon} 工具返回:</strong> <code>{tool_name}</code> - {status_text}
    </div>
    """


def chat_with_agent(message, history, tool_log):
    """
    与Agent对话的主函数
    
    Args:
        message: 用户输入
        history: 历史对话
        tool_log: 工具调用日志（不再使用，保留参数兼容性）
    
    Yields:
        tuple: (历史对话, 当前回复, 工具日志)
    """
    if not message.strip():
        return history, "", ""
    
    # 添加用户消息到历史 - 使用字典格式
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    
    # 记录上一条助手回复，用于避免第二次提问时先显示上一次回复
    previous_ai_response = ""
    if len(history) > 2:
        for msg in reversed(history[:-2]):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                previous_ai_response = msg.get("content", "") or ""
                break
    
    current_response = ""
    seen_tool_calls = set()  # 记录已显示的工具调用，避免重复
    tool_calls = []  # 收集所有工具调用信息
    
    # 定义需要显示的工具列表，排除内部工具
    valid_tools = {'get_stock_code_by_name', 'analyze_stock_trend_detailed', 'get_valid_stock_data'}
    
    try:
        # 流式处理Agent响应
        for event in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=context,
            stream_mode="values"
        ):
            # 检测工具调用
            if "messages" in event:
                messages = event["messages"]
                for msg in messages:
                    # 检测是否有工具调用
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_id = tool_call.get('id', '')
                            tool_name = tool_call.get('name', 'unknown')
                            
                            # 只显示有效的工具，过滤掉ResponseFormat等内部工具
                            if tool_name not in valid_tools:
                                continue
                            
                            # 使用工具ID避免重复显示
                            if tool_id and tool_id not in seen_tool_calls:
                                tool_input = tool_call.get('args', {})
                                params_str = json.dumps(tool_input, ensure_ascii=False)
                                
                                # 添加到工具调用列表
                                tool_calls.append({
                                    'id': tool_id,
                                    'name': tool_name,
                                    'params': params_str,
                                    'status': None
                                })
                                
                                seen_tool_calls.add(tool_id)
                    
                    # 检测工具返回结果
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        msg_id = getattr(msg, 'tool_call_id', '') or getattr(msg, 'id', '')
                        tool_name = getattr(msg, 'name', 'unknown')
                        
                        # 只处理有效的工具
                        if tool_name not in valid_tools:
                            continue
                        
                        # 使用消息ID避免重复显示
                        if msg_id and f"result_{msg_id}" not in seen_tool_calls:
                            content = getattr(msg, 'content', '')
                            
                            # 检测是否有error字段来判断成功或失败
                            success = True
                            if isinstance(content, str):
                                try:
                                    content_dict = json.loads(content)
                                    if isinstance(content_dict, dict) and 'error' in content_dict:
                                        success = False
                                except:
                                    pass
                            elif isinstance(content, dict) and 'error' in content:
                                success = False
                            
                            # 更新对应工具的状态
                            for tool in tool_calls:
                                if tool['id'] == msg_id:
                                    tool['status'] = success
                                    break
                            
                            seen_tool_calls.add(f"result_{msg_id}")
            
            # 获取AI的回复内容
            if "structured_response" in event:
                structured = event["structured_response"]
                if hasattr(structured, 'response'):
                    new_text = structured.response or ""
                    # 如果第一轮流式结果与上一条回复完全相同，则跳过，避免“重播”上一条回答
                    if not current_response and previous_ai_response and new_text == previous_ai_response:
                        continue
                    # 构建工具调用折叠部分
                    tool_section = ""
                    if tool_calls:
                        tool_items = []
                        for tool in tool_calls:
                            status = "✅ 成功" if tool['status'] else ("❌ 失败" if tool['status'] is False else "⏳ 处理中")
                            params = tool.get('params', '')
                            if params:
                                tool_items.append(f"• **{tool['name']}** - {status}\n  参数: `{params}`")
                            else:
                                tool_items.append(f"• **{tool['name']}** - {status}")
                        
                        tool_list = "\n".join(tool_items)
                        tool_section = f"""<details>
<summary>🔧 工具调用记录 ({len(tool_calls)})</summary>

{tool_list}

</details>

---

"""
                    
                    # 如果有工具调用信息，先显示，然后是AI回复
                    if tool_section:
                        current_response = tool_section + new_text
                    else:
                        current_response = new_text
                    
                    history[-1]["content"] = current_response
                    yield history, current_response, ""
                    
                    # 添加交易建议和风险提示
                    extra_info = ""
                    if hasattr(structured, 'trading_decision') and structured.trading_decision:
                        extra_info += f"\n\n📊 **交易建议:** {structured.trading_decision}"
                    if hasattr(structured, 'risk_warning') and structured.risk_warning:
                        extra_info += f"\n\n⚠️ **风险提示:** {structured.risk_warning}"
                    
                    if extra_info:
                        current_response += extra_info
                        history[-1]["content"] = current_response
                        yield history, current_response, ""
    
    except Exception as e:
        error_msg = f"❌ 发生错误: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, error_msg, ""
    
    return history, current_response, ""


def clear_conversation():
    """清空对话"""
    return [], "", ""


# 创建Gradio界面
with gr.Blocks(title="股票分析AI助手") as demo:
    gr.Markdown("""
    # 🤖 股票分析AI助手
    
    专业的股票交易助手，基于 LangChain + DeepSeek 构建
    
    **功能特点:**
    - ✅ 支持股票名称和代码查询
    - ✅ 实时技术指标分析
    - ✅ 智能交易建议
    - ✅ 工具调用实时显示
    
    **初始资金:** 30万元 | **交易单位:** 1手(100股)
    """)

    # 自定义样式：历史信息栏最小宽度 + 回车发送
    gr.HTML("""
    <style>
    #chat_history {
        min-width: 480px;
    }
    </style>
    <script>
    window.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            const active = document.activeElement;
            if (active && active.id === 'msg_input') {
                e.preventDefault();
                const btn = document.getElementById('send_btn');
                if (btn) { btn.click(); }
            }
        }
    });
    </script>
    """)
    
    with gr.Row():
        # 对话区域（全宽）
        with gr.Column():
            chatbot = gr.Chatbot(
                label="对话历史",
                show_label=False,
                avatar_images=(None, "🤖"),
                height="auto",
                container=False,  # 去掉外层容器，避免双滚动条
                elem_id="chat_history"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您的问题，例如：帮我分析一下贵州茅台（按回车发送）",
                    lines=2,
                    scale=4,
                    show_label=False,
                    elem_id="msg_input"
                )
                send_btn = gr.Button("发送 📤", variant="primary", scale=1, elem_id="send_btn")
            
            with gr.Row():
                clear_btn = gr.Button("清空对话 🗑️", variant="secondary")
                
            gr.Markdown("""
            ### 💡 使用示例
            - "帮我分析一下贵州茅台的趋势"
            - "查询平安银行的股票代码"
            - "000001最近30天的走势如何"
            - "什么板块适合投资"
            """)
    
    # 隐藏的状态组件和工具日志（保持兼容性）
    current_response = gr.State("")
    tool_log = gr.State("")
    last_user_msg = gr.State("")  # 记录上一次用户消息，避免重复显示
    
    # 事件绑定 - 回车键发送
    def handle_submit(user_msg, history, tool_log_state, last_msg):
        """处理用户提交，避免重复显示上一次回复"""
        # 如果是同一条消息，不重复处理
        if user_msg == last_msg:
            return history, "", tool_log_state, user_msg
        
        # 调用聊天函数
        for h, resp, tl in chat_with_agent(user_msg, history, tool_log_state):
            yield h, resp, tl, user_msg
    
    send_event = msg_input.submit(
        handle_submit,
        inputs=[msg_input, chatbot, tool_log, last_user_msg],
        outputs=[chatbot, current_response, tool_log, last_user_msg]
    )
    
    send_event.then(
        lambda: "",
        outputs=[msg_input]
    )
    
    send_btn_event = send_btn.click(
        handle_submit,
        inputs=[msg_input, chatbot, tool_log, last_user_msg],
        outputs=[chatbot, current_response, tool_log, last_user_msg]
    )
    
    send_btn_event.then(
        lambda: "",
        outputs=[msg_input]
    )
    
    clear_btn.click(
        lambda: ([], "", "", ""),
        outputs=[chatbot, current_response, tool_log, last_user_msg]
    )
    
    gr.Markdown("""
    ---
    <div style='text-align: center; color: #666;'>
        <small>⚠️ 投资有风险，建议仅供参考 | Powered by LangChain & DeepSeek</small>
    </div>
    """)


if __name__ == "__main__":
    print("🚀 启动股票分析AI助手...")
    print("📍 访问地址: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
