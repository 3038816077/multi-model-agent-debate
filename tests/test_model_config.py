"""模型配置一致性测试。

这些用例守的是「配置漂移」：
代码里引用的模型名必须真的在注册表里，否则运行时会静默退化，
比如分层路由全都匹配不上，自动路由和分歧分析就悄悄掉到最便宜档。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_model_max_tokens, get_model_registry, resolve_model_id  # noqa: E402
from app.core.router import MODEL_PRICE_PER_M, MODEL_TIERS, has_price  # noqa: E402


def test_every_tier_candidate_exists_in_registry():
    """分层里写的每个名字都必须能在注册表找到——删模型时最容易漏改这里。"""
    registry = get_model_registry()
    if not registry:
        return  # 没配注册表时跳过（CI 里用假配置）
    missing = []
    for tier, candidates in MODEL_TIERS.items():
        for name in candidates:
            if name not in registry:
                missing.append((tier, name))
    assert not missing, f"分层引用了不存在的模型：{missing}"


def test_registry_entries_have_required_fields():
    for key, cfg in (get_model_registry() or {}).items():
        assert cfg.get("base_url"), f"{key} 缺 base_url"
        assert cfg.get("label"), f"{key} 缺 label（前端会显示成 key）"


def test_resolve_model_id_prefers_explicit_model():
    registry = get_model_registry()
    for key, cfg in (registry or {}).items():
        real = resolve_model_id(key)
        assert real, f"{key} 解析出的模型名为空"
        if cfg.get("model"):
            assert real == cfg["model"]


def test_max_tokens_roundtrip():
    assert get_model_max_tokens("不存在的模型", 2048) == 2048
    registry = get_model_registry()
    for key, cfg in (registry or {}).items():
        if cfg.get("max_tokens"):
            assert get_model_max_tokens(key) == int(cfg["max_tokens"])


def test_price_table_keys_are_resolvable():
    """价格表的 key 要么是真实模型名，要么能被注册表解析到。"""
    registry = get_model_registry()
    known = set()
    for key, cfg in (registry or {}).items():
        known.add(key)
        known.add(cfg.get("model") or key)
    stale = [name for name in MODEL_PRICE_PER_M if known and name not in known]
    # 允许保留历史模型的价格（不影响统计），但不能全是废条目
    assert len(stale) < len(MODEL_PRICE_PER_M), "价格表里没有任何一个当前模型"


def test_has_price_marks_unpriced_models():
    assert has_price("绝对不存在的模型-zzz") is False
    registry = get_model_registry()
    for key, cfg in (registry or {}).items():
        real = cfg.get("model") or key
        expected = (real in MODEL_PRICE_PER_M) or (key in MODEL_PRICE_PER_M)
        assert has_price(key) is expected, key


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
