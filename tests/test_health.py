"""健康检查接口的结构测试。

不依赖模型服务：只验证返回结构和状态码语义（致命问题必须 503）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.health import health_payload  # noqa: E402


def test_health_shape():
    code, body = health_payload()
    assert code in (200, 503)
    assert body["status"] in ("ok", "degraded", "error")
    assert isinstance(body["models"], int)
    assert isinstance(body["tiers"], dict)
    assert isinstance(body["problems"], list)


def test_health_problems_are_structured():
    _, body = health_payload()
    for p in body["problems"]:
        assert p.get("level") in ("fatal", "warn", "info"), p
        assert p.get("code"), p
        assert p.get("message"), p
    levels = [p["level"] for p in body["problems"]]
    # 有致命问题时必须返回 503，不能假装健康
    code, _ = health_payload()
    assert (code == 503) == ("fatal" in levels)


def test_health_reports_something_when_unconfigured():
    """没配注册表时必须明确报出来，而不是返回 ok。"""
    code, body = health_payload()
    if body["models"] == 0:
        assert code == 503
        assert any(p["code"] == "registry_empty" for p in body["problems"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
