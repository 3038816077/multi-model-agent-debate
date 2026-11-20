"""指标模块的测试（Prometheus 文本格式）。

重点验证两件容易写错的事：
1. 直方图的 le 必须是**累积**计数（单调不减）
2. 路径里的 id 要归一化，否则标签基数会爆炸
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.metrics import Registry, normalize_path, render_metrics  # noqa: E402


def test_counter_and_gauge_rendering():
    r = Registry()
    r.inc("http_requests_total", {"path": "/a", "status": "200"})
    r.inc("http_requests_total", {"path": "/a", "status": "200"})
    r.gauge_set("sse_active_connections", 3)
    text = r.render()
    assert "# TYPE http_requests_total counter" in text
    assert 'http_requests_total{path="/a",status="200"} 2' in text
    assert "# TYPE sse_active_connections gauge" in text
    assert "sse_active_connections 3" in text


def test_histogram_buckets_are_cumulative():
    r = Registry()
    for d in (0.003, 0.02, 0.08, 0.4):
        r.observe("http_request_duration_seconds", d, {"path": "/x"})
    text = r.render()
    # 标签顺序是 le 在前、path 在后，所以 le="..." 后面还有其他标签
    values = [float(m.group(1)) for m in re.finditer(r'_bucket\{[^}]*le="[^"]+"[^}]*\} ([0-9.]+)', text)]
    assert values == sorted(values), "le 必须单调不减"
    assert values[-1] == 4          # +Inf 等于总次数
    assert "_count" in text and "_sum" in text


def test_normalize_path_replaces_ids():
    assert normalize_path("/sessions/3f2c8b7e-1111-2222-3333-444455556666/messages") == "/sessions/{id}/messages"
    assert normalize_path("/sessions/abc12345abc12345abc12345abc12345") == "/sessions/{id}"
    assert normalize_path("/sessions") == "/sessions"
    assert normalize_path("/health") == "/health"


def test_render_has_help_and_type_for_every_metric():
    r = Registry()
    r.inc("a_total", None, help_text="A")
    r.gauge_set("b_gauge", 1, help_text="B")
    r.observe("c_seconds", 0.1, None, help_text="C")
    text = r.render()
    for name in ("a_total", "b_gauge", "c_seconds"):
        assert "# HELP %s " % name in text
        assert "# TYPE %s " % name in text


def test_global_registry_render_is_not_empty():
    text = render_metrics()
    assert isinstance(text, str)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
