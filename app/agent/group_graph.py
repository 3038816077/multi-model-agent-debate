import concurrent.futures
from typing import TypedDict, List, Dict
from openai import OpenAI
from langgraph.graph import StateGraph, END
from app.core.config import get_model_extra_body, get_model_registry, resolve_model_id, LLM_TIMEOUT, LLM_MAX_RETRIES
from app.core.logger import logger

class GroupState(TypedDict):
    query: str
    chief_model: str
    staff_models: List[str]
    first_round: Dict[str, str]  # { "qwen-plus": "我的观点...", "glm-4-flash": "..." }
    discussion: Dict[str, str]   # { "qwen-plus": "我补充...", "glm-4-flash": "..." }
    final_summary: str
    failed_models: List[str]

def get_client_for_model(model_name: str):
    """从注册表获取指定模型的客户端"""
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"模型 {model_name} 不在注册表中")
    config = registry[model_name]
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"],
                  timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)

def _extra_kwargs(model_name: str) -> dict:
    """注册表里给该模型配了 extra_body 就带上（比如豆包关掉深度思考）。"""
    extra = get_model_extra_body(model_name)
    return {"extra_body": extra} if extra else {}


# ==================== 节点 1：副模型首轮独立作答 ====================
def staff_first_round(state: GroupState):
    query = state["query"]
    staff_models = state["staff_models"]
    results = {}
    failed = []

    def call_model(model_name):
        try:
            client = get_client_for_model(model_name)
            prompt = f"你是一名专家，请针对以下问题，给出你独立、专业的见解。\n问题：{query}"
            resp = client.chat.completions.create(
                model=resolve_model_id(model_name),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                **_extra_kwargs(model_name)
            )
            return model_name, resp.choices[0].message.content
        except Exception as e:
            logger.error(f"模型 {model_name} 首轮作答失败: {e}")
            return model_name, None

    logger.info(f"开始第一轮独立作答，参与模型: {staff_models}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(staff_models)) as executor:
        futures = [executor.submit(call_model, m) for m in staff_models]
        for future in concurrent.futures.as_completed(futures):
            m_name, content = future.result()
            if content:
                results[m_name] = content
            else:
                failed.append(m_name)

    return {"first_round": results, "failed_models": failed}

# ==================== 节点 2：副模型交叉研讨 ====================
def staff_discussion(state: GroupState):
    query = state["query"]
    first_round = state["first_round"]
    results = {}

    # 把第一轮成功的结论拼装成上下文
    context_str = ""
    for m, ans in first_round.items():
        context_str += f"【{m} 的观点】：\n{ans}\n\n"

    def call_discuss(model_name):
        try:
            client = get_client_for_model(model_name)
            prompt = f"针对问题：{query}\n\n其他专家的第一轮观点如下：\n{context_str}\n\n请你基于以上信息，发表你的补充意见，或者指出他们可能存在的漏洞。请保持专业客观。"
            resp = client.chat.completions.create(
                model=resolve_model_id(model_name),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                **_extra_kwargs(model_name)
            )
            return model_name, resp.choices[0].message.content
        except Exception as e:
            logger.error(f"模型 {model_name} 讨论阶段失败: {e}")
            return model_name, None

    logger.info("开始第二轮交叉研讨...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(first_round)) as executor:
        futures = [executor.submit(call_discuss, m) for m in first_round.keys()]
        for future in concurrent.futures.as_completed(futures):
            m_name, content = future.result()
            if content:
                results[m_name] = content

    return {"discussion": results}

# ==================== 节点 3：主模型终极总结 ====================
def chief_summary(state: GroupState):
    query = state["query"]
    chief_model = state["chief_model"]
    first_round = state["first_round"]
    discussion = state["discussion"]

    logger.info(f"主模型 {chief_model} 开始进行总结...")

    prompt = f"用户提出了以下问题：{query}\n\n"
    prompt += "=== 第一阶段：专家独立观点 ===\n"
    for m, ans in first_round.items():
        prompt += f"[{m}]: {ans}\n"
    
    prompt += "\n=== 第二阶段：专家交叉研讨 ===\n"
    for m, ans in discussion.items():
        prompt += f"[{m}]: {ans}\n"

    prompt += "\n作为首席总结官，请你综合以上所有专家的意见，去粗取精，剔除错误信息，给用户输出一份逻辑清晰、结构完整的最终回答。不要提及具体专家的名字，直接以最终结论的形式输出。请使用 Markdown 格式。"

    try:
        client = get_client_for_model(chief_model)
        resp = client.chat.completions.create(
            model=resolve_model_id(chief_model),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            **_extra_kwargs(model_name)
        )
        summary = resp.choices[0].message.content
    except Exception as e:
        logger.error(f"主模型 {chief_model} 总结失败: {e}")
        summary = "由于主模型接口异常，本次总结失败。请查看上方各位专家的原始回答。"

    return {"final_summary": summary}

# ==================== 组装 LangGraph ====================
workflow = StateGraph(GroupState)

workflow.add_node("staff_first_round", staff_first_round)
workflow.add_node("staff_discussion", staff_discussion)
workflow.add_node("chief_summary", chief_summary)

workflow.set_entry_point("staff_first_round")
workflow.add_edge("staff_first_round", "staff_discussion")
workflow.add_edge("staff_discussion", "chief_summary")
workflow.add_edge("chief_summary", END)

# 编译 Graph
group_app = workflow.compile()