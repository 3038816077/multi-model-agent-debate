"""模型分层路由：按任务复杂度挑模型，而不是所有请求都走最贵的那个。

核心假设
--------
真正需要强模型的请求是少数：
  - 代码 / 架构 / 长文分析 / 带文档的问答
而大量请求其实是体力活：
  - 翻译、摘要、格式转换、短问答

把这些体力活交给免费或便宜的模型，强模型只处理需要推理的部分。

为什么用规则分类而不是模型分类
------------------------------
调用一次模型来"判断该用哪个模型"本身就要花钱和 300ms+ 延迟，
而且会让链路更难解释。规则分类 0 延迟、0 成本、结果可复现，
覆盖大部分日常请求；拿不准时一律返回 None（不接管），
宁可多花钱，也不要答错。
"""

import contextvars
import json
import os
import re
from typing import Optional

from app.core.config import get_model_registry, USAGE_LOG_PATH
from app.core.logger import logger

TIER_CHEAP = "cheap"

# 当前请求命中的档位，跟着请求上下文走，写用量日志时就不用再传一遍
_tier_ctx = contextvars.ContextVar("router_tier", default=None)
TIER_MID = "mid"
TIER_STRONG = "strong"

# 每一档的候选模型，从前往后取第一个在注册表里真实存在的。
# 换模型只需要改这里，不用动业务代码。
MODEL_TIERS = {
    TIER_CHEAP: ["glm-4-flash", "doubao-seed-2-1-lite", "qwen-plus"],
    TIER_MID: ["deepseek-chat", "kimi-k3.1-pro", "glm-4-flash"],
    TIER_STRONG: ["gpt-6.1-sol", "claude-opus-4-6", "gpt-6", "gpt-5.6", "claude-sonnet-4-6"],
}

# 每百万 token 的参考单价（美元，input, output），只用于统计成本。
# 价格为公开价的大致值，请按自己实际账单调整；不认识的模型按 0 计。
MODEL_PRICE_PER_M = {
    "glm-4-flash": (0.00, 0.00),
    "qwen-plus": (0.40, 1.20),
    "doubao-seed-2-1-lite": (0.15, 0.30),
    "deepseek-chat": (0.27, 1.10),
    "kimi-k3.1-pro": (0.60, 2.50),
    "gpt-5.6": (1.25, 10.00),
    "gpt-6": (2.50, 15.00),
    "gpt-6.1-sol": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
}

# 需要强模型：代码、推理、方案设计、长文分析
_STRONG_HINTS = re.compile(
    r"代码|编程|报错|异常|堆栈|调试|debug|重构|架构|设计|实现|源码|算法|"
    r"选型|对比|方案|分析|推理|证明|为什么|原理|优化"
)

# 纯文本变换：翻译、改写、格式转换。这一类不需要任何工具，最安全。
_PURE_CHEAP = re.compile(
    r"翻译|译成|译成中文|改写|润色|格式化|转换成|转为|缩写|精简一下"
)

# 轻体力活：摘要、提炼、归纳。可能夹带工具需求，排在工具判断之后。
_WEAK_CHEAP = re.compile(
    r"摘要|总结一下|总结一下|提炼|概括|归纳|列一下|列个|整理成|提取"
)

# 明确的工具动作：这类请求依赖 function calling，弱模型容易调错参数，直接不接管
_TOOL_ACTION = re.compile(
    r"搜索|搜一下|搜下|查一下|查下|联网|上网查|最新资讯|最新动态|最新消息"
)

# 工具型话题：天气、时间。命中也交给当前模型，避免弱模型不会调工具
_TOOL_DOMAIN = re.compile(
    r"天气|气温|温度|下雨|下雪|刮风|几点|日期|星期几|几号"
)


def classify(message: str, has_doc: bool = False, web_search: bool = False) -> Optional[str]:
    """判断该走哪一档；返回 None 表示不接管，继续用当前/默认模型。"""
    text = (message or "").strip()
    if not text:
        return None

    # 联网搜索要走工具调用，交给用户当前选的模型，不猜
    if web_search:
        return None

    # 带文档的问答需要长上下文理解能力
    if has_doc:
        return TIER_STRONG

    # 代码 / 推理 / 方案：强模型
    if _STRONG_HINTS.search(text) or len(text) >= 300:
        return TIER_STRONG

    # 明确的工具动作（搜索、联网）：弱模型不会调 function calling，不接管
    if _TOOL_ACTION.search(text):
        return None

    # 纯文本变换（翻译、改写、格式化）：不需要工具，放心给便宜模型
    if _PURE_CHEAP.search(text):
        return TIER_CHEAP

    # 工具型话题（天气、时间）：可能触发工具调用，不接管
    if _TOOL_DOMAIN.search(text):
        return None

    # 摘要 / 提炼这类轻体力活
    if _WEAK_CHEAP.search(text):
        return TIER_CHEAP

    # 很短的问句，通常是打个招呼或者随口一问
    if len(text) <= 15:
        return TIER_CHEAP

    return TIER_MID


def set_current_tier(tier: Optional[str]) -> None:
    """把本轮命中的档位记进上下文（asyncio.to_thread 会带着它进子线程）。"""
    _tier_ctx.set(tier)


def get_current_tier() -> Optional[str]:
    return _tier_ctx.get()


def pick_model(tier: str) -> Optional[str]:
    """从该档位的候选里挑一个注册表中真实存在的模型。"""
    registry = get_model_registry()
    for name in MODEL_TIERS.get(tier, []):
        if name in registry:
            return name
    return None


def route(message: str, has_doc: bool = False, web_search: bool = False):
    """一次调用拿到 (档位, 模型)；不做任何副作用，方便单测和离线评测。"""
    tier = classify(message, has_doc=has_doc, web_search=web_search)
    if tier is None:
        return None, None
    return tier, pick_model(tier)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按价格表估算这次调用的美元成本。"""
    price_in, price_out = MODEL_PRICE_PER_M.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def log_usage(trace_id: str, model: str, prompt_tokens: int, completion_tokens: int,
              latency_ms: Optional[int] = None, tier: Optional[str] = None) -> None:
    """把每次模型调用追加到 JSONL，用于事后算成本 / 做对照实验。

    只追加、不读取、失败不影响主流程。
    """
    try:
        record = {
            "trace_id": trace_id,
            "model": model,
            "tier": tier or get_current_tier(),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "latency_ms": latency_ms,
            "cost_usd": round(estimate_cost(model, prompt_tokens or 0, completion_tokens or 0), 8),
        }
        directory = os.path.dirname(USAGE_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[用量统计] 写入失败（不影响主流程）：{e}")
