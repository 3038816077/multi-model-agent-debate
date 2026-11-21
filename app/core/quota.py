"""按天计的调用额度：公开演示时防止被当成免费 API。

设计要点（三个容易踩的坑都有对应处理）
--------------------------------------
1. 跨天重置：用本地日期做 key，请求进来时懒重置，不需要定时任务
2. 并发丢计数：所有读写走同一把锁，先加后判，不会超发
3. 伪造来源：默认只信 TCP 对端地址；X-Forwarded-For 只在 TRUST_PROXY=1 时看，
   且取**最后一段**（最靠近我们的那层代理写的），第一段是客户端随便填的
"""
import os
import threading
import time


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class DailyQuota:
    """按 (日期, 身份) 计数；超出上限拒绝。"""

    def __init__(self, per_identity: int = 30, global_limit: int = 300):
        self.per_identity = max(1, per_identity)
        self.global_limit = max(1, global_limit)
        self._lock = threading.Lock()
        self._day = ""
        self._counts = {}
        self._total = 0

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def _rollover(self):
        today = self._today()
        if today != self._day:
            self._day = today
            self._counts = {}
            self._total = 0

    def check_and_consume(self, identity: str):
        """返回 (是否放行, 该身份剩余次数, 该身份上限)。先加后判，并发不超发。"""
        with self._lock:
            self._rollover()
            used = self._counts.get(identity, 0)
            if used >= self.per_identity or self._total >= self.global_limit:
                return False, 0, self.per_identity
            self._counts[identity] = used + 1
            self._total += 1
            return True, self.per_identity - (used + 1), self.per_identity

    def snapshot(self) -> dict:
        with self._lock:
            self._rollover()
            return {
                "day": self._day,
                "total": self._total,
                "global_limit": self.global_limit,
                "per_identity": self.per_identity,
                "identities": dict(self._counts),
            }

    def reset(self):
        with self._lock:
            self._day = self._today()
            self._counts = {}
            self._total = 0


QUOTA = DailyQuota(
    per_identity=_env_int("DAILY_QUOTA_PER_IP", 30),
    global_limit=_env_int("DAILY_QUOTA_GLOBAL", 300),
)


def quota_enabled() -> bool:
    return os.getenv("DAILY_QUOTA_ENABLED", "0") not in ("0", "false", "False", "")


def client_ip(request) -> str:
    """默认只信 TCP 对端；TRUST_PROXY=1 时取 X-Forwarded-For 的最后一段。"""
    if os.getenv("TRUST_PROXY", "0") in ("1", "true", "True"):
        forwarded = (request.headers.get("x-forwarded-for") or "").strip()
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            if parts:
                return parts[-1]
    return getattr(getattr(request, "client", None), "host", "") or "unknown"


def is_owner(request) -> bool:
    """持有服务端 API_KEY 的请求视为管理员，不计额度。"""
    expected = os.getenv("API_KEY")
    provided = request.headers.get("x-api-key")
    return bool(expected and provided and provided == expected)
