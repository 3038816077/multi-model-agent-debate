import json
import time
import concurrent.futures
from app.core.logger import logger, get_trace, set_trace
from app.tools.registry import TOOL_SCHEMAS, execute_tool
from app.agent.state import State
from app.core.config import get_model_extra_body
from app.core.llm_factory import get_llm_client, get_model_name, get_llm_context, set_llm_context, friendly_llm_error
from app.core.router import log_usage

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

    _t0 = time.time()
    try:
        _extra = get_model_extra_body(get_model_name())
        _kw = {"extra_body": _extra} if _extra else {}
        resp = get_llm_client().chat.completions.create(
            model=get_model_name(),
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
            **_kw
        )
    except Exception as e:
        # 抛出中文可读的原因，前端【服务异常：...】里能直接看到该改什么
        raise RuntimeError(friendly_llm_error(e, get_model_name())) from e
    # 记录用量：算成本、做对照实验都靠这份数据
    _usage = getattr(resp, "usage", None)
    if _usage is not None:
        log_usage(
            trace_id=get_trace(),
            model=get_model_name(),
            prompt_tokens=getattr(_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(_usage, "completion_tokens", 0) or 0,
            latency_ms=int((time.time() - _t0) * 1000),
        )

    msg = resp.choices[0].message
    logger.info(f"[LLM返回] 内容: {msg.content}, 工具调用: {msg.tool_calls}")
    return {"messages": messages + [msg]}

def tool_node(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    session_id = state.get("session_id", None)
    retry_count = state.get("retry_count", 0)
    
    if not getattr(last_message, "tool_calls", None):
        return {"messages": messages}
    
    tool_results = []
    has_error = False
    
    current_trace_id = get_trace()
    current_llm_client, current_model_name = get_llm_context()

    def run_single_tool(tool_call):
        set_trace(current_trace_id)
        set_llm_context(current_llm_client, current_model_name)
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except Exception:
            tool_args = {}
        content = execute_tool(tool_call.function.name, tool_args, session_id)
        return tool_call, content

    max_workers = min(len(last_message.tool_calls), 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_tool, tc) for tc in last_message.tool_calls]
        for future in futures:
            tool_call, content = future.result()
            if content.startswith(("错误：", "工具执行异常：", "查询失败：", "知识库检索失败：")):
                has_error = True
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content
            })
    
    new_messages = messages + tool_results
    
    if has_error:
        if retry_count < 2:
            logger.warning(f"工具报错，触发大模型自愈重试 (第 {retry_count + 1} 次)")
            new_messages.append({"role": "system", "content": "上一次工具调用报错了，请仔细检查你的参数，修正后重新调用工具。"})
            return {"messages": new_messages, "retry_count": retry_count + 1}
        else:
            logger.error("工具连续报错达到上限，停止重试")
            new_messages.append({"role": "system", "content": "工具已多次报错，请不要再次调用工具，直接如实告知用户工具暂时不可用，并给出友好提示。"})
            return {"messages": new_messages, "retry_count": retry_count}

    return {"messages": new_messages, "retry_count": 0}