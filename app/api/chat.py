import copy, uuid, re
import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

import app.tools.weather
import app.tools.knowledge
from app.core.config import MODEL_NAME, MAX_HISTORY_LENGTH
from app.core.logger import logger, set_trace
from app.core.utils import clean_latex_text, clean_weather_reply, generate_dialog_summary, extract_long_term_memory
from app.tools.calc import get_current_time, calculate_expression
from app.tools.registry import WEATHER_CACHE
from app.agent.graph import agent_app

# 👇 导入数据库模块
from app.db.database import SessionLocal
from app.db.crud import get_messages, save_message, get_or_create_session, update_long_term_memory

# 👇 导入鉴权和限流组件
from app.middleware.auth import limiter, verify_api_key

# 👇 导入 LLM 动态工厂
from app.core.llm_factory import set_llm_client, get_llm_client, mask_api_key

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_api_key: Optional[str] = None
    user_base_url: Optional[str] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

system_prompt = r"""【身份】你是严格执行规则的智能助手，所有计算、查询必须调用工具，禁止编造答案。
... (保持你原来的 prompt 不变) ...
"""

# ==================== 1. 同步对话接口 ====================
@router.post("/chat", summary="对话接口")
@limiter.limit("30/minute")
def chat(request: Request, request_body: ChatRequest, db: Session = Depends(get_db), auth: bool = Depends(verify_api_key)):
    current_trace_id = set_trace()
    if request_body.user_api_key:
        logger.info(f"[自定义Key] 使用用户 Key: {mask_api_key(request_body.user_api_key)}")
        set_llm_client(request_body.user_api_key, request_body.user_base_url)

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

    logger.info("【最终用户输入】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    save_message(db, session_id, "user", request_body.message)

    if is_time_question:
        real_time = get_current_time()
        messages.append({"role": "system", "content": f"实时时间结果：{real_time}。请基于这个结果用自然口语回答用户。"})
        resp = get_llm_client().chat.completions.create(model=MODEL_NAME, messages=messages)
        final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
        save_message(db, session_id, "assistant", final_answer)
        return {"code": 200, "answer": final_answer, "session_id": session_id, "trace_id": current_trace_id}

    if is_calc_question and not is_time_question:
        expr_match = re.search(r'[\d+\-*/().\s]+', request_body.message)
        expression = expr_match.group().strip() if expr_match else request_body.message
        calc_result = calculate_expression(expression)
        messages.append({"role": "system", "content": f"工具计算结果：{calc_result}。请仅基于这个数学结果用自然语言回答，附带计算过程。"})
        resp = get_llm_client().chat.completions.create(model=MODEL_NAME, messages=messages)
        final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
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
        set_llm_client(request_body.user_api_key, request_body.user_base_url)

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

    logger.info("【最终用户输入(流式)】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    save_message(db, session_id, "user", request_body.message)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'event': 'thinking', 'data': '正在思考...'})}\n\n"
            await asyncio.sleep(0.1)

            # ========== 时间拦截（流式） ==========
            if is_time_question:
                real_time = get_current_time()
                messages.append({"role": "system", "content": f"实时时间结果：{real_time}。请基于这个结果用自然口语回答用户。"})
                resp = get_llm_client().chat.completions.create(model=MODEL_NAME, messages=messages)
                final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
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

            # ========== 计算拦截（流式） ==========
            if is_calc_question and not is_time_question:
                expr_match = re.search(r'[\d+\-*/().\s]+', request_body.message)
                expression = expr_match.group().strip() if expr_match else request_body.message
                calc_result = calculate_expression(expression)
                messages.append({"role": "system", "content": f"工具计算结果：{calc_result}。请仅基于这个数学结果用自然语言回答，附带计算过程，不要发散。"})
                resp = get_llm_client().chat.completions.create(model=MODEL_NAME, messages=messages)
                final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
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

            # ========== Agent 逻辑（流式） ==========
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