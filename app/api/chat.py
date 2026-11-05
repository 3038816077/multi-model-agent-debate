import copy, uuid, re
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

import app.tools.weather
import app.tools.knowledge
from app.core.config import client, MODEL_NAME, MAX_HISTORY_LENGTH
from app.core.logger import logger
from app.core.utils import clean_latex_text, clean_weather_reply, generate_dialog_summary, extract_long_term_memory
from app.tools.calc import get_current_time, calculate_expression
from app.tools.registry import WEATHER_CACHE
from app.agent.graph import agent_app

# 👇 导入数据库模块
from app.db.database import SessionLocal
from app.db.crud import get_messages, save_message, get_or_create_session, update_long_term_memory

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# 👇 依赖注入，自动管理数据库连接的打开与关闭
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

system_prompt = r"""【身份】你是严格执行规则的智能助手，所有计算、查询必须调用工具，禁止编造答案。
... (保持你的原版 prompt 完全不变) ...
"""

@router.post("/chat", summary="对话接口")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id or str(uuid.uuid4())
    
    # 1. 从数据库加载历史消息（只取最近的 MAX_HISTORY_LENGTH 条）
    messages = get_messages(db, session_id, limit=MAX_HISTORY_LENGTH)
    
    if not messages:
        # 如果是新会话，初始化并存入 system prompt
        messages = [{"role": "system", "content": system_prompt}]
        save_message(db, session_id, "system", system_prompt)

    # 2. 从数据库加载长期记忆
    db_session = get_or_create_session(db, session_id)
    memory_text = db_session.long_term_memory
    if memory_text:
        messages.insert(1, {"role": "system", "content": f"【用户长期记忆信息】\n{memory_text}\n\n以上是用户的长期固定信息，可以直接引用，无需重复询问用户。"})

    # 3. 意图判断与路由控制
    time_keywords = ["时间", "几点", "日期", "星期几", "今天几号", "现在什么时候"]
    is_time_question = any(keyword in request.message for keyword in time_keywords)
    
    calc_pattern = re.compile(r'计算|等于多少|算式|表达式|\d+\s*[+\-*/()]\s*\d+')
    is_calc_question = bool(calc_pattern.search(request.message))
    
    weather_keywords = ["天气", "温度", "下雨", "下雪", "冷", "热", "度", "风", "雾", "霾"]
    is_weather_question = any(kw in request.message for kw in weather_keywords) or "呢" in request.message
    
    # 注入天气缓存
    known_info = []
    if not is_time_question and not is_calc_question and session_id in WEATHER_CACHE:
        for city_weather in WEATHER_CACHE[session_id].values():
            known_info.append("已知信息：" + city_weather)
    
    user_content = request.message
    if known_info:
        user_content = "内部参考数据：\n" + "\n".join(known_info) + "\n\n用户问题：" + request.message + "\n\n要求：1. 参考数据直接使用。2. 省略问法默认查天气，直接调用工具。3. 绝对禁止复述参考数据，禁止提已知信息，禁止问是否重新查询，直接输出简洁结果。"

    logger.info("【最终用户输入】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    
    # 👇 将用户原始提问存入数据库（注意存的是 request.message，而不是带参考数据的 user_content）
    save_message(db, session_id, "user", request.message)

    # ========== 时间拦截 ==========
    if is_time_question:
        real_time = get_current_time()
        messages.append({"role": "system", "content": f"实时时间结果：{real_time}。请基于这个结果用自然口语回答用户，不要提及工具、规则、已知信息，直接给出清晰答案。"})
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
        
        # 👇 存 AI 回复并更新长期记忆
        save_message(db, session_id, "assistant", final_answer)
        try:
            new_memory = extract_long_term_memory(messages, memory_text)
            update_long_term_memory(db, session_id, new_memory)
        except Exception as e:
            logger.warning(f"长期记忆更新失败：{str(e)}")
            
        return {"code": 200, "answer": final_answer, "session_id": session_id, "history_length": len(messages)}

    # ========== 计算拦截 ==========
    if is_calc_question and not is_time_question:
        expr_match = re.search(r'[\d+\-*/().\s]+', request.message)
        expression = expr_match.group().strip() if expr_match else request.message
        calc_result = calculate_expression(expression)
        messages.append({"role": "system", "content": f"工具计算结果：{calc_result}。请仅基于这个数学结果用自然语言回答，附带计算过程，不要发散，不要涉及任何天气或无关信息。"})
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages)
        final_answer = clean_weather_reply(clean_latex_text(resp.choices[0].message.content))
        
        thought_process = f"【执行链路】\n\n第1步 调用工具：calculate_expression\n参数：{{\"expression\": \"{expression}\"}}\n\n第2步 返回结果：{calc_result}"
        
        # 👇 存 AI 回复（带思考过程）并更新长期记忆
        save_message(db, session_id, "assistant", final_answer, thought_process)
        try:
            new_memory = extract_long_term_memory(messages, memory_text)
            update_long_term_memory(db, session_id, new_memory)
        except Exception as e:
            logger.warning(f"长期记忆更新失败：{str(e)}")
            
        return {"code": 200, "answer": final_answer, "session_id": session_id, "history_length": len(messages), "thought_process": thought_process}
        
    # ========== Agent 逻辑 ==========
    try:
        input_len = len(messages)
        result = agent_app.invoke({"messages": messages, "session_id": session_id})
        final_messages = result["messages"]
        final_answer = clean_weather_reply(clean_latex_text(final_messages[-1].content))
        
        # 更新天气缓存
        for msg in final_messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
            name = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if role == "tool" and name == "get_weather":
                city = content.split("天气：")[0]
                if session_id not in WEATHER_CACHE:
                    WEATHER_CACHE[session_id] = {}
                WEATHER_CACHE[session_id][city] = content

        # 提取思考过程
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
        elif is_weather_question and known_info:
            thought_process = "【执行链路】\n第1步 命中本地天气缓存\n第2步 直接读取缓存数据进行回答\n\n（注：相同城市在短时间内查询，系统会自动使用缓存以降低延迟和节省API调用额度）"

        # 👇 存 AI 回复（带思考过程）并更新长期记忆
        save_message(db, session_id, "assistant", final_answer, thought_process)
        try:
            new_memory = extract_long_term_memory(final_messages, memory_text)
            update_long_term_memory(db, session_id, new_memory)
        except Exception as e:
            logger.warning(f"长期记忆更新失败：{str(e)}")

        return {"code": 200, "answer": final_answer, "session_id": session_id, "history_length": len(messages), "thought_process": thought_process}
    except Exception as e:
        logger.error(f"对话接口异常：{str(e)}", exc_info=True)
        return {"code": 500, "error": str(e)}