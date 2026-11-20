"""极简 Prometheus 指标（零依赖）。

为什么不装 prometheus_client
----------------------------
这台机器 pip 装不了包（用户目录权限被限制），但指标必须能用。
所以按 Prometheus 的**文本暴露格式**自己实现最小的 counter / gauge / histogram——
Prometheus 抓的是文本，不关心你是哪个库生成的。

约定：所有指标名带 _total / _seconds 后缀，标签值必须已归一化
（比如 path 用路由模板而不是真实 id，否则基数会爆炸）。
"""
import threading
import time

# 延迟分桶（秒）：覆盖 10ms ~ 10s
DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_LOCK = threading.RLock()


class Registry:
    def __init__(self):
        self.counters = {}    # name -> {labels: value}
        self.gauges = {}      # name -> {labels: value}
        self.histograms = {}  # name -> {labels: {"buckets":[..], "sum":x, "count":n}}
        self.meta = {}        # name -> (type, help)

    # ---------- 写入 ----------
    def inc(self, name: str, labels: dict = None, amount: float = 1.0, help_text: str = ""):
        key = tuple(sorted((labels or {}).items()))
        with _LOCK:
            self.counters.setdefault(name, {}).setdefault(key, 0.0)
            self.counters[name][key] += amount
            self.meta.setdefault(name, ("counter", help_text))

    def gauge_set(self, name: str, value: float, labels: dict = None, help_text: str = ""):
        key = tuple(sorted((labels or {}).items()))
        with _LOCK:
            self.gauges.setdefault(name, {})[key] = float(value)
            self.meta.setdefault(name, ("gauge", help_text))

    def gauge_inc(self, name: str, labels: dict = None, amount: float = 1.0, help_text: str = ""):
        key = tuple(sorted((labels or {}).items()))
        with _LOCK:
            current = self.gauges.setdefault(name, {}).get(key, 0.0)
            self.gauges[name][key] = current + amount
            self.meta.setdefault(name, ("gauge", help_text))

    def observe(self, name: str, value: float, labels: dict = None, help_text: str = ""):
        key = tuple(sorted((labels or {}).items()))
        with _LOCK:
            entry = self.histograms.setdefault(name, {}).setdefault(
                key, {"buckets": [0] * (len(DEFAULT_BUCKETS) + 1), "sum": 0.0, "count": 0}
            )
            for i, edge in enumerate(DEFAULT_BUCKETS):
                if value <= edge:
                    entry["buckets"][i] += 1
                    break
            else:
                entry["buckets"][-1] += 1
            entry["sum"] += value
            entry["count"] += 1
            self.meta.setdefault(name, ("histogram", help_text))

    # ---------- 渲染 ----------
    @staticmethod
    def _labels(key) -> str:
        if not key:
            return ""
        inner = ",".join('%s="%s"' % (k, str(v).replace('"', "'")) for k, v in key)
        return "{" + inner + "}"

    def render(self) -> str:
        lines = []
        with _LOCK:
            for name, series in sorted(self.counters.items()):
                kind, help_text = self.meta.get(name, ("counter", ""))
                lines.append("# HELP %s %s" % (name, help_text or name))
                lines.append("# TYPE %s counter" % name)
                for key, value in sorted(series.items()):
                    lines.append("%s%s %s" % (name, self._labels(key), _num(value)))

            for name, series in sorted(self.gauges.items()):
                kind, help_text = self.meta.get(name, ("gauge", ""))
                lines.append("# HELP %s %s" % (name, help_text or name))
                lines.append("# TYPE %s gauge" % name)
                for key, value in sorted(series.items()):
                    lines.append("%s%s %s" % (name, self._labels(key), _num(value)))

            for name, series in sorted(self.histograms.items()):
                kind, help_text = self.meta.get(name, ("histogram", ""))
                lines.append("# HELP %s %s" % (name, help_text or name))
                lines.append("# TYPE %s histogram" % name)
                for key, entry in sorted(series.items()):
                    base = dict(key)
                    # Prometheus 的 le 是「小于等于该值的累计数」，必须单调不减
                    cumulative = 0
                    for i, edge in enumerate(DEFAULT_BUCKETS):
                        cumulative += entry["buckets"][i]
                        labels = dict(base, le=_num(edge))
                        lines.append("%s_bucket%s %s" % (name, self._labels(tuple(sorted(labels.items()))), cumulative))
                    cumulative += entry["buckets"][-1]
                    labels = dict(base, le="+Inf")
                    lines.append("%s_bucket%s %s" % (name, self._labels(tuple(sorted(labels.items()))), cumulative))
                    lines.append("%s_sum%s %s" % (name, self._labels(key), _num(entry["sum"])))
                    lines.append("%s_count%s %s" % (name, self._labels(key), entry["count"]))
        return "\n".join(lines) + "\n"

    def reset(self):
        with _LOCK:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.meta.clear()


def _num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return ("%.6f" % value).rstrip("0").rstrip(".")


# 全局注册表
METRICS = Registry()


_UUID_RE = None


def normalize_path(path: str) -> str:
    """把 /sessions/3f2c.../messages 归一成 /sessions/{id}/messages。

    不然每个会话 id 都会变成一个独立标签，Prometheus 时序基数会爆炸。
    """
    global _UUID_RE
    if _UUID_RE is None:
        import re as _re

        _UUID_RE = _re.compile(r"/(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32}|\d+)(?=/|$)")
    return _UUID_RE.sub("/{id}", path)


def record_http(path: str, status: int, duration: float):
    """记录一次 HTTP 请求。path 用路由模板，避免标签基数爆炸。"""
    path = normalize_path(path)
    METRICS.inc("http_requests_total", {"path": path, "status": str(status)},
                help_text="Total HTTP requests")
    METRICS.observe("http_request_duration_seconds", duration, {"path": path},
                    help_text="HTTP request duration in seconds")


def record_llm(model: str, scene: str, tokens: int = 0):
    METRICS.inc("llm_calls_total", {"model": model, "scene": scene},
                help_text="LLM call count")
    if tokens:
        METRICS.inc("llm_tokens_total", {"model": model}, amount=tokens,
                    help_text="Cumulative tokens")


def render_metrics() -> str:
    return METRICS.render()
