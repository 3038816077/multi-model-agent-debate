"""多模型并行模式的共享类型与客户端工厂。

历史说明：这个模块原本是一个 LangGraph 三阶段流程（独立作答 → 交叉互评 → 首席总结），
但 HTTP 接口一直是在 app/api/chat.py 里自己实现的，graph 从未被调用；而且 chief_summary 里
引用了未定义的 model_name，一旦真被调用就会 NameError。

更重要的是，60 题对照实验（见 eval/REPORT.md）显示「首席总结」这一步在扣分：

    并行 − 单模型 = +2.50  95% CI [+1.92, +3.08]
    研讨 − 单模型 = +1.63  95% CI [+1.13, +2.14]
    研讨 − 并行   = −0.87  95% CI [−1.29, −0.44]   ← 显著更差，成本还翻倍

所以现在只保留"多个模型独立作答 + 分歧分析"，把互评与总结都去掉了。
"""

from typing import Dict, List, TypedDict
from openai import OpenAI

from app.core.config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
    get_model_registry,
)


class GroupState(TypedDict):
    """多模型并行的运行状态。"""

    query: str
    chief_model: str
    staff_models: List[str]
    first_round: Dict[str, str]  # { "gpt-5.6": "我的观点...", "kimi-k3.1-pro": "..." }
    failed_models: List[str]


def get_client_for_model(model_name: str) -> OpenAI:
    """从注册表取指定模型的客户端。"""
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(f"模型 {model_name} 不在注册表中")
    config = registry[model_name]
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )
