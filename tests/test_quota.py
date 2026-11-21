"""每日额度的单元测试。

守四件事：按身份隔离、达到上限拒绝、跨天重置、并发不超发。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.quota import DailyQuota, client_ip, is_owner  # noqa: E402


def test_allows_until_limit():
    q = DailyQuota(per_identity=3, global_limit=100)
    for i in range(3):
        ok, remaining, _ = q.check_and_consume("1.1.1.1")
        assert ok is True
        assert remaining == 3 - (i + 1)
    ok, remaining, _ = q.check_and_consume("1.1.1.1")
    assert ok is False and remaining == 0


def test_identities_are_isolated():
    q = DailyQuota(per_identity=1, global_limit=100)
    assert q.check_and_consume("a")[0] is True
    assert q.check_and_consume("b")[0] is True
    assert q.check_and_consume("a")[0] is False


def test_global_limit_blocks_everyone():
    q = DailyQuota(per_identity=100, global_limit=2)
    assert q.check_and_consume("a")[0] is True
    assert q.check_and_consume("b")[0] is True
    assert q.check_and_consume("c")[0] is False


def test_day_rollover_resets_counts():
    q = DailyQuota(per_identity=1, global_limit=10)
    assert q.check_and_consume("a")[0] is True
    assert q.check_and_consume("a")[0] is False
    # 手动把「今天」改成昨天，模拟跨天
    q._day = "2000-01-01"
    q._rollover()
    assert q.check_and_consume("a")[0] is True
    assert q.snapshot()["total"] == 1


def test_concurrent_consume_never_exceeds_limit():
    q = DailyQuota(per_identity=50, global_limit=1000)
    results = []
    lock = threading.Lock()

    def worker():
        ok, _, _ = q.check_and_consume("same-ip")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(120)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert sum(1 for r in results if r) == 50
    assert q.snapshot()["identities"]["same-ip"] == 50


def test_snapshot_shape():
    q = DailyQuota(per_identity=2, global_limit=10)
    q.check_and_consume("x")
    snap = q.snapshot()
    assert set(snap) == {"day", "total", "global_limit", "per_identity", "identities"}
    assert snap["total"] == 1


def test_client_ip_ignores_forwarded_header_by_default():
    class Req:
        headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        client = type("C", (), {"host": "9.9.9.9"})()

    os.environ.pop("TRUST_PROXY", None)
    assert client_ip(Req()) == "9.9.9.9"


def test_client_ip_uses_last_hop_when_trusted():
    class Req:
        headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        client = type("C", (), {"host": "9.9.9.9"})()

    os.environ["TRUST_PROXY"] = "1"
    try:
        # 第一段是客户端可伪造的，只信最后一段
        assert client_ip(Req()) == "5.6.7.8"
    finally:
        os.environ.pop("TRUST_PROXY", None)


def test_owner_key_bypasses():
    class Req:
        headers = {"x-api-key": "secret-123"}

    os.environ["API_KEY"] = "secret-123"
    try:
        assert is_owner(Req()) is True
        assert is_owner(type("R", (), {"headers": {"x-api-key": "wrong"}})()) is False
    finally:
        os.environ.pop("API_KEY", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
