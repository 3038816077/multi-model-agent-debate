import copy, uuid, re
import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.core.config import MAX_HISTORY_LENGTH
import app.tools.weather
import app.tools.knowledge
import app.tools.web_search # 确保注册了联网工具
from app.core.llm_factory import set_llm_client, get_llm_client, mask_api_key, get_model_name
from app.core.logger import logger, set_trace
from app.core.utils import clean_latex_text, clean_weather_reply, generate_dialog_summary, extract_long_term_memory
from app.tools.calc import get_current_time, calculate_expression
from app.tools.registry import WEATHER_CACHE
from app.agent.graph import agent_app

from app.db.database import SessionLocal
from app.db.crud import get_messages, save_message, get_or_create_session, update_long_term_memory
from app.middleware.auth import limiter, verify_api_key

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_api_key: Optional[str] = None
    user_base_url: Optional[str] = None
    user_model_name: Optional[str] = None
    enable_web_search: Optional[bool] = False # 是否开启强制联网

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 👇 这里是之前丢失的完整 system_prompt，已经修复了 Markdown 冲突
system_prompt = r"""【身份】你是严格执行规则的智能助手，所有计算、查询必须调用工具，禁止编造答案。
【最高优先级规则】
0. 如果用户询问的是“你自己”具备什么功能、能做什么，请直接用自然语言回答（如查天气、算术、查时间）。如果用户询问的是“特定产品”的使用说明、操作规则、外部文档等知识，必须调用 search_knowledge 检索。
1. 你的所有输出必须使用标准 Markdown 格式（如加粗、列表、表格、代码块），但绝对禁止使用 LaTeX 数学公式（如 \times、\frac 等）。
2. 天气查询、数值计算类问题，必须立刻调用对应工具获取结果，绝对禁止在未调用工具的情况下自行编造或输出任何半截结果。
3. 对话中注入的参考数据仅作内部使用，回答时禁止复述、禁止提及，禁止询问用户是否需要重新查询，直接输出核心结果即可。
4. 支持多工具并行调用，同类型多任务必须一次性批量执行，禁止逐轮单工具调用。

【强制规则】
1. 开头的「已知信息」里已有的数据，直接使用，禁止重复调用工具、禁止反问用户。
2. 缺少必要信息时才调用工具；多个任务必须一次性调用全部工具。
3. 数值计算必须调用 calculate_expression 工具，禁止心算直接给结果。
4. 工具出错先修正参数重试1次，仍失败再如实告知原因。
5. 计算类问题返回结果时，必须用自然语言给出明确结论，再附带计算过程。禁止只输出算式、禁止只说“计算结果正确”。
6. 最终回答必须贴合用户的问题场景，直接给出用户想要的答案，不要站在工具校验的角度回复。
7. 不向用户展示工具调用过程。
8. 调用知识库工具后，只能基于检索到的内容回答，禁止编造知识库外的信息；知识库没有相关内容时，如实告知用户「暂未查询到相关信息」，禁止胡编；回答时不要提及「知识库」「检索」等内部逻辑。
9. 对话中提供的上下文参考内容，仅作为内部数据使用，回答时直接引用结论，不要提及「上下文」「已知信息」等表述，也不要复述前置内容。
10. 当你获取到天气工具的结果后，请直接以自然语言输出，可以包含「XX天气：」的格式，但不要在没拿到数据时强行输出该前缀。
11. 当用户询问“你有什么功能”、“你能做什么”等自身能力问题时，请直接列出你的技能列表（查天气、计算、时间、知识库检索、联网搜索）。绝对禁止引用历史对话中的具体查询结果作为举例。
12. 当你调用了 search_knowledge 工具后，必须基于检索结果回答，并且在回答中明确标注信息来源。格式要求：根据【来源文件: xxx.txt】的内容，……。绝对禁止编造来源信息。

【可用工具】
- calculate_expression：通用数学计算，传入表达式字符串
- get_current_time：获取当前时间，无需参数
- get_weather：查询城市天气，传入城市名称
- search_knowledge：从本地知识库检索信息
- search_web：联网搜索最新资讯
"""

# ==================== 1. 同步对话接口 ====================
@router.post("/chat", summary="对话接口")
@limiter.limit("30/minute")
def chat(request: Request, request_body: ChatRequest, db: Session = Depends(get_db), auth: bool = Depends(verify_api_key)):
    current_trace_id = set_trace()
    if request_body.user_api_key:
        logger.info(f"[自定义Key] 使用用户 Key: {mask_api_key(request_body.user_api_key)}")
        set_llm_client(request_body.user_api_key, request_body.user_base_url, request_body.user_model_name)

    session_id = request_body.session_id or str(uuid.uuid4())
    messages = get_messages(db, session_id, limit=MAX_HISTORY_LENGTH)
    if not messages:
        messages = [{"role": "system", "content": system_prompt}]
        save_message(db, session_id, "system", system_prompt)

    db_session = get_or_create_session(db, session_id)
    memory_text = db_session.long_term_memory
    if memory_text:
        messages.insert(1, {"role": "system", "content": f"【用户长期记忆信息】\n{memory_text}\n\n以上是用户的长期固定信息，可以直接引用，无需重复询问用户。"})

    time_keywords = ["时间", "几点", "日期", "星期几", "今天几号", "现在什么时候"]
    is_time_question = any(keyword in request_body.message for keyword in time_keywords)
    calc_pattern = re.compile(r'计算|等于多少|算式|表达式|\d+\s*[+\-*/()]\s*\d+')
    is_calc_question = bool(calc_pattern.search(request_body.message))
    weather_keywords = ["天气", "温度", "下雨", "下雪", "冷", "热", "度", "风", "雾", "霾"]
    is_weather_question = any(kw in request_body.message for kw in weather_keywords) or "呢" in request_body.message
    
    known_info = []
    if is_weather_question and not is_time_question and not is_calc_question and session_id in WEATHER_CACHE:
        for city_weather in WEATHER_CACHE[session_id].values():
            known_info.append("已知信息：" + city_weather)
    
    user_content = request_body.message
    if known_info:
        user_content = "内部参考数据：\n" + "\n".join(known_info) + "\n\n用户问题：" + request_body.message + "\n\n要求：1. 参考数据直接使用。2. 省略问法默认查天气，直接调用工具。3. 绝对禁止复述参考数据，禁止提已知信息，禁止问是否重新查询，直接输出简洁结果。"
    
    if request_body.enable_web_search:
        logger.info("[强制联网] 用户开启了联网搜索开关")
        user_content = f"【系统强制指令】用户已开启联网搜索，请务必优先调用 search_web 工具查询最新资讯。\n{user_content}"

    logger.info("【最终用户输入】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    save_message(db, session_id, "user", request_body.message)

    if is_time_question:
        real_time = get_current_time()
        final_answer = f"当前时间是：{real_time}。"
        save_message(db, session_id, "assistant", final_answer)
        return {"code": 200, "answer": final_answer, "session_id": session_id, "trace_id": current_trace_id}

    if is_calc_question and not is_time_question:
        expr_match = re.search(r'[\d+\-*/().\s]+', request_body.message)
        expression = expr_match.group().strip() if expr_match else request_body.message
        calc_result = calculate_expression(expression)
        final_answer = f"{calc_result}。"
        thought_process = f"【执行链路】\n\n第1步 调用工具：calculate_expression\n参数：{{\"expression\": \"{expression}\"}}\n\n第2步 返回结果：{calc_result}"
        save_message(db, session_id, "assistant", final_answer, thought_process)
        return {"code": 200, "answer": final_answer, "session_id": session_id, "thought_process": thought_process, "trace_id": current_trace_id}
        
    try:
        input_len = len(messages)
        result = agent_app.invoke({"messages": messages, "session_id": session_id, "retry_count": 0})
        final_messages = result["messages"]
        final_answer = clean_weather_reply(clean_latex_text(final_messages[-1].content))
        for msg in final_messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "tool" and name == "get_weather":
                city = content.split("天气：")[0]
                if session_id not in WEATHER_CACHE: WEATHER_CACHE[session_id] = {}
                WEATHER_CACHE[session_id][city] = content

        thought_process = ""
        step_list = []
        step_index = 1
        new_messages = final_messages[input_len:]
        for msg in new_messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                        name = func.get("name") if isinstance(func, dict) else getattr(func, "name", "")
                        args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                        step_list.append(f"第{step_index}步 调用工具：{name}\n参数：{args}")
                        step_index += 1
            elif role == "tool":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                step_list.append(f"第{step_index}步 返回结果：{content}")
                step_index += 1

        if step_list:
            thought_process = "【执行链路】\n" + "\n\n".join(step_list)
            if "search_knowledge" in thought_process and "【来源文件:" in thought_process:
                sources = re.findall(r'【来源文件: (.*?)】', thought_process)
                if sources:
                    unique_sources = list(dict.fromkeys(sources))
                    source_str = "、".join(unique_sources)
                    if "来源" not in final_answer:
                        final_answer += f"\n\n(来源: {source_str})"
        elif is_weather_question and known_info:
            thought_process = "【执行链路】\n第1步 命中本地天气缓存\n第2步 直接读取缓存数据进行回答"

        save_message(db, session_id, "assistant", final_answer, thought_process)
        return {"code": 200, "answer": final_answer, "session_id": session_id, "thought_process": thought_process, "trace_id": current_trace_id}
    except Exception as e:
        logger.error(f"对话接口异常：{str(e)}", exc_info=True)
        return {"code": 500, "error": str(e)}

# ==================== 2. SSE 流式对话接口 ====================
@router.post("/chat/stream", summary="流式对话接口")
@limiter.limit("30/minute")
async def chat_stream(request: Request, request_body: ChatRequest, db: Session = Depends(get_db), auth: bool = Depends(verify_api_key)):
    current_trace_id = set_trace()
    if request_body.user_api_key:
        logger.info(f"[自定义Key] 使用用户 Key: {mask_api_key(request_body.user_api_key)}")
        set_llm_client(request_body.user_api_key, request_body.user_base_url, request_body.user_model_name)

    session_id = request_body.session_id or str(uuid.uuid4())
    messages = get_messages(db, session_id, limit=MAX_HISTORY_LENGTH)
    if not messages:
        messages = [{"role": "system", "content": system_prompt}]
        save_message(db, session_id, "system", system_prompt)

    db_session = get_or_create_session(db, session_id)
    memory_text = db_session.long_term_memory
    if memory_text:
        messages.insert(1, {"role": "system", "content": f"【用户长期记忆信息】\n{memory_text}\n\n以上是用户的长期固定信息，可以直接引用，无需重复询问用户。"})

    time_keywords = ["时间", "几点", "日期", "星期几", "今天几号", "现在什么时候"]
    is_time_question = any(keyword in request_body.message for keyword in time_keywords)
    calc_pattern = re.compile(r'计算|等于多少|算式|表达式|\d+\s*[+\-*/()]\s*\d+')
    is_calc_question = bool(calc_pattern.search(request_body.message))
    weather_keywords = ["天气", "温度", "下雨", "下雪", "冷", "热", "度", "风", "雾", "霾"]
    is_weather_question = any(kw in request_body.message for kw in weather_keywords) or "呢" in request_body.message
    
    known_info = []
    if is_weather_question and not is_time_question and not is_calc_question and session_id in WEATHER_CACHE:
        for city_weather in WEATHER_CACHE[session_id].values():
            known_info.append("已知信息：" + city_weather)
    
    user_content = request_body.message
    if known_info:
        user_content = "内部参考数据：\n" + "\n".join(known_info) + "\n\n用户问题：" + request_body.message + "\n\n要求：1. 参考数据直接使用。2. 省略问法默认查天气，直接调用工具。3. 绝对禁止复述参考数据，禁止提已知信息，禁止问是否重新查询，直接输出简洁结果。"

    if request_body.enable_web_search:
        logger.info("[强制联网] 用户开启了联网搜索开关 (流式)")
        user_content = f"【系统强制指令】用户已开启联网搜索，请务必优先调用 search_web 工具查询最新资讯。\n{user_content}"

    logger.info("【最终用户输入(流式)】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    save_message(db, session_id, "user", request_body.message)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'event': 'thinking', 'data': '正在思考...'})}\n\n"
            await asyncio.sleep(0.1)

            if is_time_question:
                real_time = get_current_time()
                final_answer = f"当前时间是：{real_time}。"
                thought_process = f"【执行链路】\n\n第1步 获取实时时间\n\n第2步 返回结果：{real_time}"
                save_message(db, session_id, "assistant", final_answer, thought_process)
                yield f"data: {json.dumps({'event': 'thinking_start'}, ensure_ascii=False)}\n\n"
                for char in thought_process:
                    yield f"data: {json.dumps({'event': 'thinking', 'data': char}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)
                yield f"data: {json.dumps({'event': 'thinking_done'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)
                for char in final_answer:
                    yield f"data: {json.dumps({'event': 'content', 'data': char}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.03)
                yield f"data: {json.dumps({'event': 'done', 'trace_id': current_trace_id, 'session_id': session_id}, ensure_ascii=False)}\n\n"
                return

            if is_calc_question and not is_time_question:
                expr_match = re.search(r'[\d+\-*/().\s]+', request_body.message)
                expression = expr_match.group().strip() if expr_match else request_body.message
                calc_result = calculate_expression(expression)
                final_answer = f"{calc_result}。"
                thought_process = f"【执行链路】\n\n第1步 调用工具：calculate_expression\n参数：{{\"expression\": \"{expression}\"}}\n\n第2步 返回结果：{calc_result}"
                save_message(db, session_id, "assistant", final_answer, thought_process)
                yield f"data: {json.dumps({'event': 'thinking_start'}, ensure_ascii=False)}\n\n"
                for char in thought_process:
                    yield f"data: {json.dumps({'event': 'thinking', 'data': char}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)
                yield f"data: {json.dumps({'event': 'thinking_done'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)
                for char in final_answer:
                    yield f"data: {json.dumps({'event': 'content', 'data': char}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.03)
                yield f"data: {json.dumps({'event': 'done', 'trace_id': current_trace_id, 'session_id': session_id}, ensure_ascii=False)}\n\n"
                return

            input_len = len(messages)
            result = agent_app.invoke({"messages": messages, "session_id": session_id, "retry_count": 0})
            final_messages = result["messages"]
            final_answer = clean_weather_reply(clean_latex_text(final_messages[-1].content))
            
            thought_process = ""
            step_list = []
            step_index = 1
            new_messages = final_messages[input_len:]
            for msg in new_messages:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "assistant":
                    tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                            name = func.get("name") if isinstance(func, dict) else getattr(func, "name", "")
                            args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                            step_list.append(f"第{step_index}步 调用工具：{name}\n参数：{args}")
                            step_index += 1
                elif role == "tool":
                    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    step_list.append(f"第{step_index}步 返回结果：{content}")
                    step_index += 1

            if step_list:
                thought_process = "【执行链路】\n" + "\n\n".join(step_list)
                if "search_knowledge" in thought_process and "【来源文件:" in thought_process:
                    sources = re.findall(r'【来源文件: (.*?)】', thought_process)
                    if sources:
                        unique_sources = list(dict.fromkeys(sources))
                        source_str = "、".join(unique_sources)
                        if "来源" not in final_answer:
                            final_answer += f"\n\n(来源: {source_str})"
            elif is_weather_question and known_info:
                thought_process = "【执行链路】\n第1步 命中本地天气缓存\n第2步 直接读取缓存数据进行回答"

            for msg in final_messages:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role == "tool" and name == "get_weather":
                    city = content.split("天气：")[0]
                    if session_id not in WEATHER_CACHE: WEATHER_CACHE[session_id] = {}
                    WEATHER_CACHE[session_id][city] = content

            save_message(db, session_id, "assistant", final_answer, thought_process)

            if thought_process:
                yield f"data: {json.dumps({'event': 'thinking_start'}, ensure_ascii=False)}\n\n"
                for char in thought_process:
                    yield f"data: {json.dumps({'event': 'thinking', 'data': char}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)
                yield f"data: {json.dumps({'event': 'thinking_done'}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)

            for char in final_answer:
                yield f"data: {json.dumps({'event': 'content', 'data': char}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.03)

            yield f"data: {json.dumps({'event': 'done', 'trace_id': current_trace_id, 'session_id': session_id}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式接口异常：{str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==================== 3. 多会话管理接口 ====================
@router.get("/sessions", summary="获取所有会话列表")
def get_sessions(db: Session = Depends(get_db)):
    from app.db.models import DBSession, DBMessage
    sessions = db.query(DBSession).order_by(DBSession.updated_at.desc()).all()
    result = []
    for s in sessions:
        first_msg = db.query(DBMessage).filter(DBMessage.session_id == s.session_id, DBMessage.role == "user").order_by(DBMessage.created_at.asc()).first()
        title = "新会话"
        if first_msg and first_msg.content:
            content_clean = first_msg.content.strip().replace("\n", " ")
            title = content_clean[:15] + ("..." if len(content_clean) > 15 else "")
        result.append({"session_id": s.session_id, "updated_at": s.updated_at.strftime("%m-%d %H:%M"), "title": title})
    return {"code": 200, "data": result}

@router.get("/sessions/{session_id}/messages", summary="获取指定会话的历史消息")
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    from app.db.models import DBMessage
    msgs = db.query(DBMessage).filter(DBMessage.session_id == session_id, DBMessage.role.in_(["user", "assistant"])).order_by(DBMessage.created_at.asc()).all()
    return {"code": 200, "data": [{"role": m.role, "content": m.content, "thought_process": m.thought_process or ""} for m in msgs]}