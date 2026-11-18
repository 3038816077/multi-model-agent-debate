"""分歧报告的回归测试。

守三件事：
1. 编造的引文必须被降级成「未核实」，不能当成真分歧展示
2. 同立场的回答必须合并成一组，不能按模型拆成多组
3. 解析要能吃下模型偶尔加的代码围栏 / 多余文字
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.chat import (  # noqa: E402
    _parse_divergence,
    _render_divergence_data,
    filter_divergence,
    render_divergence,
    verify_quote,
)

ANSWERS = [
    "高并发下应该先加缓存降低数据库压力，缓存能挡住大部分读请求。",
    "先做限流保证系统不被压垮，缓存扛不住写流量。",
    "高并发下应该先加缓存，同时配合限流。",
]
LABELS = ["豆包", "Kimi", "GLM"]


def _conflict(quote_a, quote_b):
    return {
        "consensus": [],
        "conflict": [{
            "point": "先做哪一个",
            "groups": [
                {"answers": [1, 3], "stance": "先加缓存", "from": 1, "quote": quote_a},
                {"answers": [2], "stance": "先限流", "from": 2, "quote": quote_b},
            ],
        }],
        "oneoff": [],
    }


def test_verify_quote_exact():
    assert verify_quote("高并发下应该先加缓存", ANSWERS[0]) is True


def test_verify_quote_tolerates_punctuation_change():
    assert verify_quote("高并发下应该先加缓存降低数据库压力", ANSWERS[0]) is True


def test_verify_quote_rejects_fabrication():
    assert verify_quote("缓存会在写入时自动分片并保证强一致", ANSWERS[0]) is False


def test_verify_quote_rejects_too_short():
    assert verify_quote("先加缓存", ANSWERS[0]) is False


def test_parse_strips_code_fence():
    raw = "```json\n{\"consensus\": [\"a\"], \"conflict\": [], \"oneoff\": []}\n```"
    data = _parse_divergence(raw)
    assert data["consensus"] == ["a"]


def test_parse_tolerates_prose_around_json():
    raw = "好的，结果如下：\n{\"consensus\": [\"b\"], \"conflict\": [], \"oneoff\": []}\n以上。"
    data = _parse_divergence(raw)
    assert data["consensus"] == ["b"]


def test_fabricated_quote_is_downgraded():
    data, dropped_quote, _ = filter_divergence(
        _conflict("先做限流保证系统不被压垮", "先做限流才能保证系统稳定不被打挂"),
        ANSWERS, "", "测试",
    )
    assert dropped_quote == 1
    assert data["conflict"][0]["verified"] is False


def test_verified_quote_keeps_conflict():
    data, dropped_quote, _ = filter_divergence(
        _conflict("高并发下应该先加缓存降低数据库压力", "先做限流保证系统不被压垮"),
        ANSWERS, "", "测试",
    )
    assert dropped_quote == 0
    assert data["conflict"][0]["verified"] is True


def test_render_groups_same_stance_together():
    md = render_divergence(
        json.dumps(_conflict("高并发下应该先加缓存降低数据库压力",
                             "先做限流保证系统不被压垮"), ensure_ascii=False),
        ANSWERS, "", "测试", LABELS,
    )
    assert "分歧" in md
    assert "豆包、GLM" in md
    assert "Kimi" in md


def test_render_downgraded_goes_to_unverified_section():
    md = render_divergence(
        json.dumps(_conflict("高并发下应该先加缓存降低数据库压力",
                             "先做限流才能保证系统稳定不被打挂"), ensure_ascii=False),
        ANSWERS, "", "测试", LABELS,
    )
    assert "未核实" in md


def test_render_no_conflict_says_so():
    data = {"consensus": ["都要做"], "conflict": [], "oneoff": []}
    md = _render_divergence_data(data, 0, 0, LABELS)
    assert "没有发现实质矛盾" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
