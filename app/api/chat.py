import copy, uuid, re, time
import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List

import app.tools.weather
import app.tools.knowledge
import app.tools.web_search
from app.core.config import client, MODEL_NAME, MAX_HISTORY_LENGTH, get_model_registry, resolve_model_id, get_model_extra_body, AUTO_ROUTE
from app.core.llm_factory import set_llm_client, get_llm_client, mask_api_key, get_model_name, friendly_llm_error
from app.core.router import classify, pick_model, set_current_tier, log_usage, TIER_CHEAP
from app.core.logger import logger, set_trace
from app.core.utils import clean_latex_text, clean_weather_reply, generate_dialog_summary, extract_long_term_memory
from app.tools.calc import get_current_time, calculate_expression
from app.tools.registry import WEATHER_CACHE
from app.agent.graph import agent_app
from app.agent.group_graph import GroupState, get_client_for_model

from app.db.database import SessionLocal
from app.db.crud import get_messages, save_message, get_or_create_session, update_long_term_memory
from app.middleware.auth import limiter, verify_api_key
from app.api.upload import load_document

# 注入到提示词里的文档正文上限（字），超出的部分截断并告知模型
MAX_DOC_CHARS_IN_PROMPT = 12000


def _model_extra(model_id: str) -> dict:
    """注册表里给该模型配了 extra_body 就带上（例如豆包 2.1 关掉深度思考）。"""
    extra = get_model_extra_body(model_id)
    return {"extra_body": extra} if extra else {}


def _log_resp(resp, model_id: str, trace_id: str, started_at: float) -> None:
    """把一次模型调用的用量记进 usage.jsonl；失败不影响主流程。"""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    log_usage(
        trace_id=trace_id,
        model=model_id,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_ms=int((time.time() - started_at) * 1000),
    )



def _apply_auto_route(request_body):
    """按任务复杂度自动挑模型。返回 {"tier","model"}；不接管时返回 None。

    优先级：用户显式指定模型 > 自动路由 > 系统默认模型。
    """
    if not request_body.auto_route:
        return None
    if not AUTO_ROUTE:
        # 最容易踩的坑：启动时忘了设环境变量，或者 set AUTO_ROUTE=true 带了尾随空格
        logger.warning("[auto-route] 请求要求自动路由，但服务端 AUTO_ROUTE 未开启；"
                       "启动前请设置环境变量 AUTO_ROUTE=true")
        return None
    if request_body.user_model_name or request_body.user_api_key:
        return None

    tier = classify(
        request_body.message,
        has_doc=bool(request_body.attachment_id),
        web_search=bool(request_body.enable_web_search),
    )
    if not tier:
        return None

    model = pick_model(tier)
    if not model:
        logger.warning(f"[自动路由] 档位 {tier} 在注册表里没有可用模型，保持当前模型")
        return None

    set_llm_client(model_name=model)
    set_current_tier(tier)
    logger.info(f"[auto-route] tier={tier} model={model} len={len(request_body.message)}")
    return {"tier": tier, "model": model}


def _document_message(doc_id: str):
    """把上传文档的正文包装成一条 system 消息；文档不存在返回 None。"""
    doc = load_document(doc_id)
    if not doc:
        return None
    text = doc.get("text", "")
    truncated = len(text) > MAX_DOC_CHARS_IN_PROMPT
    if truncated:
        text = text[:MAX_DOC_CHARS_IN_PROMPT]
    tail = (f"\n\n（文档较长，以上只截取了前 {MAX_DOC_CHARS_IN_PROMPT} 字；"
            "如果用户问的内容不在这里，请明确告诉用户文档被截断了）") if truncated else ""
    return {
        "role": "system",
        "content": (
            f"【用户上传的文档：{doc['filename']}（{doc.get('kind','文档')}，共 {doc.get('chars', len(text))} 字）】\n"
            f"{text}{tail}\n"
            "【文档结束】\n"
            "请优先依据以上文档内容回答；文档里没有的信息要明确说“文档中未提及”，不要编造。"
        ),
    }

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    attachment_id: Optional[str] = None
    user_api_key: Optional[str] = None
    user_base_url: Optional[str] = None
    user_model_name: Optional[str] = None
    enable_web_search: Optional[bool] = False
    # 前端传 true 才启用自动路由；显式选了模型时以用户选择为准
    auto_route: Optional[bool] = False

class GroupChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    attachment_id: Optional[str] = None
    chief_model: str
    staff_models: List[str]
    # debate  = 独立作答 → 交叉互评 → 首席总结（默认，质量最高）
    # parallel = 只并行作答 → 自动提取共识与分歧（快、省，不做互评）
    mode: str = "debate"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== 获取可用模型列表 ====================
@router.get("/models", summary="获取所有可用模型")
def list_available_models():
    registry = get_model_registry()
    model_list = []
    for model_id, config in registry.items():
        model_list.append({
            "model_id": model_id,
            "vendor": config.get("vendor", "未知"),
            "label": config.get("label", model_id)
        })
    return {"code": 200, "data": model_list}


@router.get("/models/health", summary="逐个检测注册表里的模型是否真的可用")
def models_health(auth: bool = Depends(verify_api_key)):
    """给每个模型发一次 max_tokens=1 的最小请求，返回能不能用 + 具体原因。

    用来区分"没配 Key / Key 无效 / 额度用完 / 模型名写错 / 网络不通"，
    比在聊天里一个个试要快得多。
    """
    registry = get_model_registry()

    def probe(item):
        model_id = item[0]
        try:
            conn = get_client_for_model(model_id)
            _extra = get_model_extra_body(model_id)
            conn.chat.completions.create(
                model=resolve_model_id(model_id),
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                timeout=20,
                **({"extra_body": _extra} if _extra else {})
            )
            return {"model_id": model_id, "ok": True, "detail": "可用"}
        except Exception as e:
            return {"model_id": model_id, "ok": False, "detail": friendly_llm_error(e, model_id)}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(registry))) as ex:
        data = list(ex.map(probe, registry.items()))
    return {"code": 200, "data": data}

# ==================== 强化版 system_prompt ====================
system_prompt = r"""【身份】你是严格执行规则的智能助手，所有计算、查询必须调用工具，禁止编造答案。
【最高优先级规则】
0. 如果用户询问的是“你自己”具备什么功能、能做什么，请直接用自然语言回答（如查天气、算术、查时间）。如果用户询问的是“本产品”的使用说明、功能清单、操作规则，必须调用 search_knowledge 检索；通用技术问题不要检索。
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
8. 调用知识库工具后，若检索内容确实与问题相关，只能基于检索内容回答，禁止编造；若检索结果与问题无关（例如用户问的是通用技术、编程、架构、方案设计，而知识库里只有本产品说明），不要说「暂未查询到」，应忽略该结果，直接用你自己的通用知识正常回答。回答时不要提及「知识库」「检索」「来源文件」等内部逻辑。
9. 对话中提供的上下文参考内容，仅作为内部数据使用，回答时直接引用结论，不要提及「上下文」「已知信息」等表述，也不要复述前置内容。
10. 当你获取到天气工具的结果后，请直接以自然语言输出，可以包含「XX天气：」的格式，但不要在没拿到数据时强行输出该前缀。
11. 当用户询问“你有什么功能”、“你能做什么”等自身能力问题时，请直接列出你的技能列表（查天气、计算、时间、知识库检索、联网搜索）。绝对禁止引用历史对话中的具体查询结果作为举例。
12. 只有当检索结果确实提供了问题所需的信息时，才在回答中标注来源，格式：根据【来源文件: xxx.txt】的内容，……。检索结果与问题无关时，不要标注来源，也不要提「产品说明.txt」。绝对禁止编造来源信息。
13. 只有用户询问本产品的型号、功能清单、使用说明、内部规则这类“本地文档里才有”的事实时，才调用 search_knowledge。通用的技术问题（如 LangGraph 怎么用、BM25 和向量检索怎么选、多租户权限模型怎么设计、某个报错怎么排查）属于公开知识，直接用你自己的知识回答，不需要也不应该检索知识库。

【可用工具】
- calculate_expression：通用数学计算，传入表达式字符串
- get_current_time：获取当前时间，无需参数
- get_weather：查询城市天气，传入城市名称
- search_knowledge：从本地知识库检索信息
- search_web：联网搜索最新资讯
"""
# ==================== 确定性快路径（时间 / 纯算术） ====================
# 这两个功能必须 100% 可用，不能依赖"模型是否愿意调用工具"，所以由后端兜底。
_FILLER_WORDS = [
    "帮我算一下", "帮我计算", "请帮我算", "请计算", "计算一下", "算一下", "计算",
    "等于多少", "等于几", "是多少", "得多少", "结果是多少", "请", "一下",
]
_TIME_KEYWORDS = ["几点", "现在时间", "当前时间", "什么时间", "什么时候",
                  "今天几号", "今天日期", "星期几", "周几", "日期"]


def _normalize_expr_text(text: str) -> str:
    t = text
    for w in _FILLER_WORDS:
        t = t.replace(w, " ")
    for a, b in (("×", "*"), ("÷", "/"), ("（", "("), ("）", ")"), ("＋", "+"), ("－", "-"), ("　", " ")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip().strip("？?。.!！= ").strip()


def _extract_pure_expression(message: str):
    """只有整句就是一个算术表达式时才返回表达式。
    这样"2024年有多少天"不会被人为截成 2024 而算出一个假答案。"""
    t = _normalize_expr_text(message)
    if not t or not re.search(r"\d", t):
        return None
    if not re.fullmatch(r"[0-9+\-*/(). ]+", t):
        return None
    if not re.search(r"[+\-*/]", t):
        return None
    return t


def _is_time_question(message: str) -> bool:
    if any(k in message for k in ("天气", "温度", "气温")):
        return False
    if _extract_pure_expression(message):
        return False
    return any(k in message for k in _TIME_KEYWORDS)


def _build_fast_reply(message: str):
    """时间/纯算术的确定性回答，返回 (final_answer, thought_process)；不命中返回 None"""
    if _is_time_question(message):
        raw = get_current_time()                     # 当前时间：2026年11月09日 13:35:20,星期一
        pretty = raw.replace("当前时间：", "", 1).strip()
        answer = f"当前时间是：{pretty}"
        thought = ("【执行链路】\n\n"
                   "第1步 调用工具：get_current_time\n参数：{}\n\n"
                   f"第2步 返回结果：{raw}")
        return answer, thought

    expr = _extract_pure_expression(message)
    if expr:
        raw = calculate_expression(expr)             # 计算结果：(12+28)*3 = 120
        thought = ("【执行链路】\n\n"
                   "第1步 调用工具：calculate_expression\n"
                   f"参数：{{\"expression\": \"{expr}\"}}\n\n"
                   f"第2步 返回结果：{raw}")
        return raw, thought
    return None


def _pick_final_answer(final_messages) -> str:
    """取最终回答：优先最后一条有内容的 assistant，其次最后一次工具结果，避免出现空气泡"""
    def role_of(m):
        return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")

    def content_of(m):
        c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        return c or ""

    for msg in reversed(final_messages):
        if role_of(msg) == "assistant" and content_of(msg).strip():
            return clean_weather_reply(clean_latex_text(content_of(msg)))
    for msg in reversed(final_messages):
        if role_of(msg) == "tool" and content_of(msg).strip():
            return clean_weather_reply(clean_latex_text(content_of(msg)))
    return "抱歉，这次没有生成有效回复，请重试一次。"


def _sse(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _stream_answer(thought_process: str, final_answer: str, trace_id: str, session_id: str):
    """统一的"先思考、再回答"流式帧序列（快路径与图路径共用）"""
    if thought_process:
        yield _sse({"event": "thinking_start"})
        for char in thought_process:
            yield _sse({"event": "thinking", "data": char})
            await asyncio.sleep(0.005)
        yield _sse({"event": "thinking_done"})
        await asyncio.sleep(0.2)
    for char in final_answer:
        yield _sse({"event": "content", "data": char})
        await asyncio.sleep(0.03)
    yield _sse({"event": "done", "trace_id": trace_id, "session_id": session_id})


# ==================== 1. 单模型同步对话 ====================
@router.post("/chat", summary="对话接口")
@limiter.limit("30/minute")
def chat(request: Request, request_body: ChatRequest, db: Session = Depends(get_db), auth: bool = Depends(verify_api_key)):
    current_trace_id = set_trace()
    if request_body.user_model_name or request_body.user_api_key:
        logger.info(f"[模型切换] 目标模型: {request_body.user_model_name}, 自定义Key: {bool(request_body.user_api_key)}")
        set_llm_client(request_body.user_api_key, request_body.user_base_url, request_body.user_model_name)

    routed = _apply_auto_route(request_body)

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
    
    # 上传的 Word / PDF：正文作为 system 消息插到本轮提问之前
    if request_body.attachment_id:
        doc_msg = _document_message(request_body.attachment_id)
        if doc_msg:
            messages.append(doc_msg)
        else:
            logger.warning(f"附件 {request_body.attachment_id} 不存在或已过期，本轮忽略")

    user_content = request_body.message
    if known_info:
        user_content = "内部参考数据：\n" + "\n".join(known_info) + "\n\n用户问题：" + request_body.message + "\n\n要求：1. 参考数据直接使用。2. 省略问法默认查天气，直接调用工具。3. 绝对禁止复述参考数据，禁止提已知信息，禁止问是否重新查询，直接输出简洁结果。"

    if request_body.enable_web_search:
        logger.info("[强制联网] 用户开启了联网搜索开关")
        user_content = f"【系统强制指令】用户已开启联网搜索，请务必优先调用 search_web 工具查询最新资讯。\n{user_content}"

    logger.info("【最终用户输入】\n" + user_content + "\n")
    messages.append({"role": "user", "content": user_content})
    save_message(db, session_id, "user", request_body.message)

    # 确定性快路径：时间 / 纯算术，不依赖大模型是否愿意调用工具
    fast = _build_fast_reply(request_body.message)
    if fast:
        fast_answer, fast_thought = fast
        save_message(db, session_id, "assistant", fast_answer, fast_thought)
        return {"code": 200, "answer": fast_answer, "session_id": session_id,
                "thought_process": fast_thought, "trace_id": current_trace_id}

    try:
        input_len = len(messages)
        result = agent_app.invoke({"messages": messages, "session_id": session_id, "retry_count": 0})
        final_messages = result["messages"]
        final_answer = _pick_final_answer(final_messages)
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
        try:
            new_memory = extract_long_term_memory(final_messages, memory_text)
            update_long_term_memory(db, session_id, new_memory)
        except Exception as e:
            logger.warning(f"长期记忆更新失败：{str(e)}")

        return {"code": 200, "answer": final_answer, "session_id": session_id, "history_length": len(messages), "thought_process": thought_process, "trace_id": current_trace_id, "routed_model": (routed or {}).get("model")}
    except Exception as e:
        logger.error(f"对话接口异常：{str(e)}", exc_info=True)
        return {"code": 500, "error": str(e)}


async def _update_memory_bg(session_id: str, final_messages: list, memory_text: str):
    """后台更新长期记忆：独立会话 + 独立线程，失败只记日志，绝不影响主回答。"""
    try:
        new_memory = await asyncio.to_thread(extract_long_term_memory, final_messages, memory_text)
        bg_db = SessionLocal()
        try:
            update_long_term_memory(bg_db, session_id, new_memory)
        finally:
            bg_db.close()
    except Exception as e:
        logger.warning(f"长期记忆更新失败（不影响回答）：{str(e)}")


# ==================== 2. 单模型流式对话 ====================
@router.post("/chat/stream", summary="流式对话接口")
@limiter.limit("30/minute")
async def chat_stream(request: Request, request_body: ChatRequest, db: Session = Depends(get_db), auth: bool = Depends(verify_api_key)):
    current_trace_id = set_trace()
    if request_body.user_model_name or request_body.user_api_key:
        logger.info(f"[模型切换] 目标模型: {request_body.user_model_name}, 自定义Key: {bool(request_body.user_api_key)}")
        set_llm_client(request_body.user_api_key, request_body.user_base_url, request_body.user_model_name)

    routed = _apply_auto_route(request_body)

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
    
    # 上传的 Word / PDF：正文作为 system 消息插到本轮提问之前
    if request_body.attachment_id:
        doc_msg = _document_message(request_body.attachment_id)
        if doc_msg:
            messages.append(doc_msg)
        else:
            logger.warning(f"附件 {request_body.attachment_id} 不存在或已过期，本轮忽略")

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
            if routed:
                yield f"data: {json.dumps({'event': 'routed', 'data': routed})}\n\n"
            await asyncio.sleep(0.1)

            # 确定性快路径：时间 / 纯算术，不依赖大模型是否愿意调用工具
            fast = _build_fast_reply(request_body.message)
            if fast:
                fast_answer, fast_thought = fast
                save_message(db, session_id, "assistant", fast_answer, fast_thought)
                async for frame in _stream_answer(fast_thought, fast_answer, current_trace_id, session_id):
                    yield frame
                return

            input_len = len(messages)
            # 关键：agent_app.invoke 是同步阻塞调用（内部要等大模型返回）。
            # 直接在 async 生成器里调会占住事件循环，整个服务（含别的请求）全部卡住，
            # 一旦上游网关不回包就是"服务卡死"。丢到线程池里跑，事件循环继续工作。
            result = await asyncio.to_thread(
                agent_app.invoke,
                {"messages": messages, "session_id": session_id, "retry_count": 0}
            )
            
            final_messages = result["messages"]
            final_answer = _pick_final_answer(final_messages)
            
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
                    if session_id not in WEATHER_CACHE:
                        WEATHER_CACHE[session_id] = {}
                    WEATHER_CACHE[session_id][city] = content

            save_message(db, session_id, "assistant", final_answer, thought_process)

            # 👇 长期记忆抽取是「第二次大模型调用」，而且不影响本次回答。
            #    以前这里是同步等待：答案其实已经拿到了，却要等这次调用结束
            #    （模型一慢就是 60s × 重试），前端就一直停在"正在思考..."。
            #    改成后台任务，答案立刻下发。
            asyncio.create_task(_update_memory_bg(session_id, final_messages, memory_text))

            async for frame in _stream_answer(thought_process, final_answer, current_trace_id, session_id):
                yield frame
            
        except Exception as e:
            logger.error(f"流式接口异常：{str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------- 并行模式的“分歧分析” ----------------
# 并行模式只做第一轮（各家独立作答），不再互评。
# 但用户真正想知道的是“这几份答案到底一不一样”，
# 所以额外用便宜模型做一次比对，输出共识 / 分歧 / 孤例三段。
# 这一次调用很便宜（cheap 档），比跑完整研讨省得多。

DIVERGENCE_ANSWER_MAX_CHARS = 1500   # 单份回答截断，避免 prompt 过长

GROUP_DIVERGENCE_PROMPT = """下面是 {n} 个模型对同一个问题的独立回答，请找出它们结论上真正矛盾的地方。

问题：{query}

{answers}

判断标准只有一条：**结论是否互相矛盾**。
- 说法不同但结论一致，不算分歧
- 一个说得详细、另一个没展开，不算分歧，也不算孤例
- 不要因为“某个回答没提到某一点”就判定成分歧或孤例

严格按下面三段输出，每段没有内容就写“无”：

【共识】所有回答结论一致的核心判断，一到两句
【分歧】结论互相冲突的点，格式：某问题：回答1认为X，回答2认为Y
【孤例】只有一方给出的具体事实或数字，其他回答既没支持也没否定，但会影响结论

全文不超过 200 字，不要复述原文，不要评价哪个模型更好。"""


async def analyze_divergence(query: str, answers: dict, trace_id: str = "",
                             fallback_model: str = "") -> str:
    """用便宜模型比对多份独立回答，挑出共识、分歧和孤例。"""
    if not answers:
        return ""
    judge = pick_model(TIER_CHEAP) or fallback_model
    if not judge:
        return "（当前注册表里没有可用的便宜模型，跳过分歧分析）"

    def sync_call():
        started = time.time()
        try:
            client = get_client_for_model(judge)
            body = "\n\n".join(
                "【回答%d】\n%s" % (i + 1, (ans or "")[:DIVERGENCE_ANSWER_MAX_CHARS])
                for i, ans in enumerate(answers.values())
            )
            prompt = GROUP_DIVERGENCE_PROMPT.format(n=len(answers), query=query, answers=body)
            resp = client.chat.completions.create(
                model=resolve_model_id(judge),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                **_model_extra(judge),
            )
            _log_resp(resp, judge, trace_id, started)
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"【并行模式】分歧分析失败：{e}")
            return "分歧分析失败：" + friendly_llm_error(e, judge)

    return await asyncio.to_thread(sync_call)


# ==================== 3. 多模型研讨流式接口 ====================
@router.post("/chat/group_stream", summary="多模型研讨流式接口")
@limiter.limit("10/minute")
async def chat_group_stream(request: Request, request_body: GroupChatRequest, db: Session = Depends(get_db)):
    current_trace_id = set_trace()
    # 日志里同时给出 label 和 key：key 是注册表标识，label 才是人看的模型名
    _reg = get_model_registry()

    def _tag(name: str) -> str:
        label = (_reg.get(name) or {}).get("label")
        return f"{label}({name})" if label and label != name else name

    logger.info(
        f"【研讨模式启动】主模型:{_tag(request_body.chief_model)}, "
        f"副模型:{[_tag(m) for m in request_body.staff_models]}"
    )
    
    session_id = request_body.session_id or str(uuid.uuid4())
    save_message(db, session_id, "user", request_body.message)

    # 上传的文档：整段拼进 query，三个阶段（首轮作答 / 交叉研讨 / 首席总结）都能看到原文
    group_query = request_body.message
    if request_body.attachment_id:
        doc = load_document(request_body.attachment_id)
        if doc:
            text = doc.get("text", "")
            cut = len(text) > MAX_DOC_CHARS_IN_PROMPT
            if cut:
                text = text[:MAX_DOC_CHARS_IN_PROMPT]
            tail = "\n（文档较长，以上只截取了开头部分）" if cut else ""
            group_query = (
                f"【用户上传的文档：{doc['filename']}（{doc.get('kind', '文档')}，共 {doc.get('chars', len(text))} 字）】\n"
                f"{text}{tail}\n【文档结束】\n\n"
                "请所有专家基于以上文档内容回答下面的问题；文档里没有的信息要明确说明“文档中未提及”，不要编造。\n\n"
                f"用户问题：{request_body.message}"
            )
            logger.info(f"【研讨模式】已注入文档 {doc['filename']}（{doc.get('chars', 0)} 字，截断={cut}）")
        else:
            logger.warning(f"附件 {request_body.attachment_id} 不存在或已过期，本轮忽略")

    if request_body.chief_model in request_body.staff_models:
        async def error_gen():
            yield f"data: {json.dumps({'event': 'error', 'data': '主模型不能和副模型重复'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def event_generator():
        try:
            state: GroupState = {
                "query": group_query,
                "chief_model": request_body.chief_model,
                "staff_models": request_body.staff_models,
                "first_round": {},
                "discussion": {},
                "final_summary": "",
                "failed_models": []
            }

            # ---------------- 阶段一：第一轮独立作答 ----------------
            yield f"data: {json.dumps({'event': 'stage', 'data': '第一轮：专家独立作答'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'init_models', 'models': request_body.staff_models}, ensure_ascii=False)}\n\n"

            async def run_first_round(model_name):
                def sync_call():
                    try:
                        client = get_client_for_model(model_name)
                        prompt = f"你是一名专家，请针对以下问题，给出你独立、专业的见解。\n问题：{state['query']}"
                        _t0 = time.time()
                        resp = client.chat.completions.create(
                            model=resolve_model_id(model_name),
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=2048,
                            **_model_extra(model_name)
                        )
                        _log_resp(resp, model_name, current_trace_id, _t0)
                        return model_name, resp.choices[0].message.content, True
                    except Exception as e:
                        return model_name, friendly_llm_error(e, model_name), False
                return await asyncio.to_thread(sync_call)

            tasks_1 = [run_first_round(m) for m in request_body.staff_models]
            for future in asyncio.as_completed(tasks_1):
                m_name, content, ok = await future
                state["first_round"][m_name] = content
                if not ok:
                    # 失败的回答只用于前端展示，不参与后续研讨（否则错误文案会被当成“观点”喂给别的模型）
                    state["failed_models"].append(m_name)
                yield f"data: {json.dumps({'event': 'speak', 'stage': 1, 'model': m_name, 'content': content}, ensure_ascii=False)}\n\n"

            # 只有成功的回答才进入后续阶段
            valid_answers = {
                m: a for m, a in state["first_round"].items()
                if m not in state["failed_models"]
            }
            if not valid_answers:
                yield f"data: {json.dumps({'event': 'error', 'data': '所有副模型均失败，无法继续。'})}\n\n"
                return

            # ---------------- 并行模式：只作答，不互评 ----------------
            # 走到这里说明第一轮已经全部返回。直接做分歧分析然后收尾，
            # 后面的第二轮、首席总结整段跳过。
            if (request_body.mode or "debate").strip().lower() == "parallel":
                yield f"data: {json.dumps({'event': 'stage', 'data': '分歧分析'}, ensure_ascii=False)}\n\n"
                divergence = await analyze_divergence(
                    state["query"], valid_answers, current_trace_id, request_body.chief_model
                )
                yield f"data: {json.dumps({'event': 'divergence', 'content': divergence}, ensure_ascii=False)}\n\n"

                for m, ans in state["first_round"].items():
                    save_message(db, session_id, "assistant", ans or "", f"GROUP_SPEAK:1:{m}")
                save_message(db, session_id, "assistant", divergence or "",
                             f"GROUP_DIVERGENCE:{request_body.chief_model}")

                yield f"data: {json.dumps({'event': 'done', 'session_id': session_id})}\n\n"
                return

            # ---------------- 阶段二：第二轮交叉讨论 ----------------
            yield f"data: {json.dumps({'event': 'stage', 'data': '第二轮：交叉研讨'}, ensure_ascii=False)}\n\n"
            active_models_2 = list(valid_answers.keys())
            yield f"data: {json.dumps({'event': 'init_models', 'models': active_models_2}, ensure_ascii=False)}\n\n"

            async def run_discussion(model_name):
                def sync_call():
                    try:
                        client = get_client_for_model(model_name)
                        context_str = ""
                        for m, ans in valid_answers.items():
                            if m != model_name:
                                context_str += f"【{m} 的观点】：\n{ans}\n\n"
                        prompt = f"针对问题：{state['query']}\n\n其他专家的第一轮观点如下：\n{context_str}\n\n请你基于以上信息，发表你的补充意见，或者指出他们可能存在的漏洞。请保持专业客观。"
                        _t0 = time.time()
                        resp = client.chat.completions.create(
                            model=resolve_model_id(model_name),
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=2048,
                            **_model_extra(model_name)
                        )
                        _log_resp(resp, model_name, current_trace_id, _t0)
                        return model_name, resp.choices[0].message.content
                    except Exception as e:
                        return model_name, friendly_llm_error(e, model_name)
                return await asyncio.to_thread(sync_call)

            tasks_2 = [run_discussion(m) for m in active_models_2]
            for future in asyncio.as_completed(tasks_2):
                m_name, content = await future
                state["discussion"][m_name] = content
                yield f"data: {json.dumps({'event': 'speak', 'stage': 2, 'model': m_name, 'content': content}, ensure_ascii=False)}\n\n"

            # ---------------- 阶段三：主模型总结 ----------------
            yield f"data: {json.dumps({'event': 'stage', 'data': '首席总结'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'init_summary', 'model': request_body.chief_model}, ensure_ascii=False)}\n\n"

            def run_summary():
                try:
                    client = get_client_for_model(request_body.chief_model)
                    prompt = f"用户提出了以下问题：{state['query']}\n\n"
                    prompt += "=== 第一阶段：专家独立观点 ===\n"
                    for m, ans in valid_answers.items():
                        prompt += f"[{m}]: {ans}\n"
                    prompt += "\n=== 第二阶段：专家交叉研讨 ===\n"
                    for m, ans in state["discussion"].items():
                        prompt += f"[{m}]: {ans}\n"
                    prompt += "\n作为首席总结官，请你综合以上所有专家的意见，去粗取精，剔除错误信息，给用户输出一份逻辑清晰、结构完整的最终回答。不要提及具体专家的名字，直接以最终结论的形式输出。请使用 Markdown 格式。"

                    _t0 = time.time()
                    resp = client.chat.completions.create(
                        model=resolve_model_id(request_body.chief_model),
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2048,
                        **_model_extra(request_body.chief_model)
                    )
                    _log_resp(resp, request_body.chief_model, current_trace_id, _t0)
                    return resp.choices[0].message.content
                except Exception as e:
                    return "总结失败: " + friendly_llm_error(e, request_body.chief_model)

            summary_content = await asyncio.to_thread(run_summary)
            state["final_summary"] = summary_content
            
            yield f"data: {json.dumps({'event': 'summary', 'model': request_body.chief_model, 'content': summary_content}, ensure_ascii=False)}\n\n"
            
            # 存库：每位专家单独一条，thought_process 里带阶段和模型名，
            # 这样重新打开会话时前端能还原成同样的头像气泡，而不是一大坨纯文本
            for m, ans in state["first_round"].items():
                save_message(db, session_id, "assistant", ans or "", f"GROUP_SPEAK:1:{m}")
            for m, ans in state["discussion"].items():
                save_message(db, session_id, "assistant", ans or "", f"GROUP_SPEAK:2:{m}")
            save_message(db, session_id, "assistant", summary_content or "",
                         f"GROUP_SUMMARY:{request_body.chief_model}")

            yield f"data: {json.dumps({'event': 'done', 'session_id': session_id})}\n\n"

        except Exception as e:
            logger.error(f"研讨流异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== 4. 多会话管理接口 ====================
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