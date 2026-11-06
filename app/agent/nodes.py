import json
from app.core.config import client, MODEL_NAME
from app.core.logger import logger
from app.tools.registry import TOOL_SCHEMAS, execute_tool
from app.agent.state import State

def llm_node(state: State):
    messages = state["messages"]
    has_tool_result = any((m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "tool" for m in messages)
    tool_choice = "auto"
    if not has_tool_result:
        last_user_content = ""
        for msg in reversed(messages):
            if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) == "user":
                last_user_content = msg.get("content", "")
                break
        if "天气" in last_user_content:
            tool_choice = "required"
            logger.info("[路由控制] 检测到首次天气查询，强制模型调用工具...")

    resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOL_SCHEMAS, tool_choice=tool_choice)
    msg = resp.choices[0].message
    logger.info(f"[LLM返回] 内容: {msg.content}, 工具调用: {msg.tool_calls}")
    return {"messages": messages + [msg]}

def tool_node(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    session_id = state.get("session_id", None)
    # 👇 获取当前的重试次数，默认为 0
    retry_count = state.get("retry_count", 0)
    
    if not getattr(last_message, "tool_calls", None):
        return {"messages": messages}
    
    tool_results = []
    has_error = False  # 👇 标记本次工具调用是否发生错误
    
    for tool_call in last_message.tool_calls:
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except Exception:
            tool_args = {}
            
        content = execute_tool(tool_call.function.name, tool_args, session_id)
        
        # 👇 检查工具返回的结果里是否包含报错关键词
        if any(key in content for key in ["错误", "异常", "失败", "超时", "not found"]):
            has_error = True
            
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": content
        })
    
    new_messages = messages + tool_results
    
    # 👇 错误自愈逻辑控制
    if has_error:
        if retry_count < 2:
            logger.warning(f"工具报错，触发大模型自愈重试 (第 {retry_count + 1} 次)")
            # 注入提示词，让大模型知道报错了，需要修正
            new_messages.append({
                "role": "system", 
                "content": "上一次工具调用报错了，请仔细检查你的参数，修正后重新调用工具。"
            })
            return {"messages": new_messages, "retry_count": retry_count + 1}
        else:
            logger.error("工具连续报错达到上限，停止重试")
            new_messages.append({
                "role": "system", 
                "content": "工具已多次报错，请不要再次调用工具，直接如实告知用户工具暂时不可用，并给出友好提示。"
            })
            return {"messages": new_messages, "retry_count": retry_count}

    # 如果一切正常，重置重试次数为 0
    return {"messages": new_messages, "retry_count": 0}