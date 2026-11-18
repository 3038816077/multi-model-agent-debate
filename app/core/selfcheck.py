"""启动自检：把"跑起来才发现"的问题挪到启动时说清楚。

设计原则
--------
- 致命问题（一个模型都不可用）启动就报错，不要等到用户点发送才 500
- 局部问题（某个模型缺 key）只警告，其余模型照常服务
- 检查函数是纯函数，不连网络、不读磁盘状态，方便写单元测试

检查结果统一成 dict：{level, code, message}
    level: "fatal" | "warn" | "info"
"""
from typing import Iterable, Mapping, Optional

# 模板占位符：用户忘了填 key 时它们会原样留在配置里
_PLACEHOLDER_TOKENS = ("replace_me", "your-key", "your_key", "填入", "xxx", "sk-xxx")


def is_placeholder_key(value: Optional[str]) -> bool:
    """判断 api_key 是不是还没填。空白也算没填。"""
    text = (value or "").strip().lower()
    if not text:
        return True
    return any(token in text for token in _PLACEHOLDER_TOKENS)


def check_registry(registry: Optional[Mapping]) -> list:
    """检查模型注册表本身能不能用。"""
    problems = []
    if not registry:
        problems.append({
            "level": "fatal",
            "code": "registry_empty",
            "message": "MODEL_REGISTRY 为空：一个模型都没配置，所有对话请求都会失败。"
                       "检查 .env 里的 MODEL_REGISTRY 是否粘贴完整（单行 JSON）。",
        })
        return problems

    usable = 0
    for key, cfg in registry.items():
        if not isinstance(cfg, Mapping):
            problems.append({
                "level": "warn", "code": "invalid_entry",
                "message": f"模型 {key} 的配置不是对象，已跳过",
            })
            continue
        if not cfg.get("base_url"):
            problems.append({
                "level": "warn", "code": "missing_base_url",
                "message": f"模型 {key} 没有 base_url，调用会失败",
            })
            continue
        if is_placeholder_key(cfg.get("api_key")):
            problems.append({
                "level": "warn", "code": "placeholder_key",
                "message": f"模型 {key} 的 api_key 还是占位符，该模型不可用",
            })
            continue
        usable += 1

    if usable == 0:
        problems.append({
            "level": "fatal", "code": "no_usable_model",
            "message": "注册表里没有一个模型填了真实 api_key，服务无法回答任何问题。",
        })
    return problems


def check_tiers(registry: Optional[Mapping], tiers: Mapping) -> list:
    """检查分层路由的候选模型是否真的存在。

    这一类错很隐蔽：模型从注册表删了，但 MODEL_TIERS 里还留着名字，
    自动路由和分歧分析就会静默退化到"最便宜档"，报告质量下降却不报错。
    """
    problems = []
    keys = set((registry or {}).keys())
    for tier, candidates in (tiers or {}).items():
        if not candidates:
            problems.append({
                "level": "warn", "code": "tier_empty",
                "message": f"分层 {tier} 没有候选模型",
            })
            continue
        if not any(c in keys for c in candidates):
            problems.append({
                "level": "warn", "code": "tier_no_match",
                "message": f"分层 {tier} 的候选 {list(candidates)} 都不在注册表里，"
                           f"该档位会退化到其它层",
            })
    return problems


def check_prices(prices: Mapping, registry: Optional[Mapping]) -> list:
    """价格表缺单价不算故障，但成本统计会偏低，必须能看见。"""
    problems = []
    if not registry:
        return problems
    registry_ids = set()
    for cfg in registry.values():
        if isinstance(cfg, Mapping):
            registry_ids.add(cfg.get("model") or "")
    missing = []
    for key, cfg in registry.items():
        if not isinstance(cfg, Mapping):
            continue
        real = cfg.get("model") or key
        if real not in prices and key not in prices:
            missing.append(key)
    if missing:
        problems.append({
            "level": "info", "code": "price_missing",
            "message": f"这些模型没有配置单价，成本统计会偏低：{missing}",
        })
    return problems


def check_writable(paths: Iterable[str]) -> list:
    """启动时确认关键目录可写，避免第一次上传/写日志才失败。"""
    import os

    problems = []
    for path in paths:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
        except Exception as e:
            problems.append({
                "level": "fatal", "code": "dir_not_writable",
                "message": f"目录不可写：{path}（{e}）",
            })
    return problems


def validate_startup() -> list:
    """把所有检查跑一遍，返回问题列表（空表示一切正常）。"""
    from app.core.config import get_model_registry
    from app.core.router import MODEL_PRICE_PER_M, MODEL_TIERS

    registry = get_model_registry()
    problems = []
    problems += check_registry(registry)
    problems += check_tiers(registry, MODEL_TIERS)
    problems += check_prices(MODEL_PRICE_PER_M, registry)
    return problems


def has_fatal(problems: Iterable[Mapping]) -> bool:
    return any(p.get("level") == "fatal" for p in problems)
