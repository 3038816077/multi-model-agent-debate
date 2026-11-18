"""启动自检的回归测试。

这些用例守的是"配置错了要在启动时说出来"，而不是等用户点发送才 500。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.selfcheck import (  # noqa: E402
    check_prices,
    check_registry,
    check_tiers,
    check_writable,
    has_fatal,
    is_placeholder_key,
)


def test_placeholder_detection():
    assert is_placeholder_key("") is True
    assert is_placeholder_key("sk-REPLACE_ME") is True
    assert is_placeholder_key("填入你的Key") is True
    assert is_placeholder_key("ark-1791f077-a2bd") is False


def test_empty_registry_is_fatal():
    problems = check_registry({})
    assert has_fatal(problems) is True
    assert any(p["code"] == "registry_empty" for p in problems)


def test_registry_all_placeholders_is_fatal():
    registry = {
        "a": {"base_url": "https://x/v1", "api_key": "sk-REPLACE_ME"},
        "b": {"base_url": "https://y/v1", "api_key": ""},
    }
    problems = check_registry(registry)
    assert has_fatal(problems) is True
    assert any(p["code"] == "no_usable_model" for p in problems)
    # 每个坏条目也应该单独提示，方便定位
    assert sum(1 for p in problems if p["code"] == "placeholder_key") == 2


def test_one_good_model_is_enough():
    registry = {
        "good": {"base_url": "https://x/v1", "api_key": "sk-real-key"},
        "bad": {"base_url": "https://y/v1", "api_key": "REPLACE_ME"},
    }
    problems = check_registry(registry)
    assert has_fatal(problems) is False
    assert any(p["code"] == "placeholder_key" for p in problems)


def test_tier_pointing_at_removed_model_is_warned():
    """模型删了但分层里还留着名字——这正是我们踩过的坑。"""
    registry = {"doubao-seed-2-1": {"base_url": "u", "api_key": "k"}}
    tiers = {"cheap": ["doubao-seed-2-1"], "strong": ["gpt-6.1-sol"]}
    problems = check_tiers(registry, tiers)
    assert any(p["code"] == "tier_no_match" for p in problems)
    assert has_fatal(problems) is False


def test_prices_missing_is_info_not_fatal():
    registry = {"m": {"base_url": "u", "api_key": "k", "model": "m-260915"}}
    problems = check_prices({"other": (1, 2)}, registry)
    assert problems and problems[0]["level"] == "info"
    assert has_fatal(problems) is False


def test_writable_dir_check(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "data", "uploads")
        assert check_writable([target]) == []
        assert os.path.isdir(target)
        # 探针文件不能留下
        assert not os.path.exists(os.path.join(target, ".write_probe"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
