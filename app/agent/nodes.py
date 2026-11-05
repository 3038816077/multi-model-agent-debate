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
    
    if not getattr(last_message, "tool_calls", None):
        return {"messages": messages}
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except Exception:
            tool_args = {}
        content = execute_tool(tool_call.function.name, tool_args, session_id)
        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": content
        })
    return {"messages": messages + tool_results}