"""评测脚本的评分逻辑测试。

评分代码本身出错，比被测模型错更可怕：
它会把"没编分歧"判成"编了分歧"，然后得出所有模型都在瞎编的结论。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from benchmark_divergence import grade  # noqa: E402


def _item(conflict, keywords=()):
    return {"id": "x", "conflict": conflict, "keywords": list(keywords)}


def test_clean_pair_with_no_conflict_marker_is_not_false_alarm():
    rendered = "**共识**\n- 都要做\n\n**分歧**\n- 没有发现实质矛盾"
    ok, verified, why = grade(_item(False), rendered)
    assert ok is True, why
    assert "无冲突" in why


def test_real_conflict_still_counted():
    rendered = "**分歧**\n- 先做哪个\n  - 甲（先缓存）"
    ok, verified, _ = grade(_item(True, ["先缓存"]), rendered)
    assert ok is True and verified is True


def test_invented_conflict_is_false_alarm():
    rendered = "**分歧**\n- 先做哪个\n  - 甲（先缓存）"
    ok, _, why = grade(_item(False), rendered)
    assert ok is False and "误报" in why


def test_missed_conflict_detected():
    rendered = "几个模型的结论没有实质冲突。"
    ok, _, why = grade(_item(True, ["先缓存"]), rendered)
    assert ok is False and "漏报" in why


def test_unverified_conflict_counts_as_detected_but_unverified():
    rendered = "**未核实的分歧**\n- 先做哪个"
    ok, verified, _ = grade(_item(True, ["先做"]), rendered)
    assert ok is True and verified is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
