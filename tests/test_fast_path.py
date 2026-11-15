"""快速通道的回归测试。

背景：`_TIME_KEYWORDS` 里曾经包含"什么时候"，导致
"多模型研讨什么时候值得用" 被误判成问时间，直接返回"当前时间是：..."。
这个测试就是为了不再犯。

跑法：
    python -m pytest tests/ -q      # 或者
    python tests/test_fast_path.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.chat import _is_time_question, _extract_pure_expression  # noqa: E402


def test_time_questions_use_fast_path():
    for q in ("现在几点了", "今天几号", "今天星期几", "现在时间是多少"):
        assert _is_time_question(q) is True, q


def test_ambiguous_when_question_does_not_use_fast_path():
    """这几条是曾经被误判的，必须走大模型。"""
    for q in (
        "多模型研讨相比单模型，什么时候值得用？给出判断标准。",
        "这个功能什么时候上线",
        "什么时间适合做模型路由",
    ):
        assert _is_time_question(q) is False, q


def test_non_time_questions_do_not_use_fast_path():
    for q in ("2024年有多少天", "今天天气怎么样", "TCP 为什么三次握手"):
        assert _is_time_question(q) is False, q


def test_pure_expression_detection():
    assert _extract_pure_expression("1+2*3") == "1+2*3"
    assert _extract_pure_expression("(12+28)*3 等于多少") == "(12+28)*3"
    # 带文字的不该被当成纯算式
    assert _extract_pure_expression("2024年有多少天") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
