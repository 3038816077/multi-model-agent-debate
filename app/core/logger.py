import logging
import contextvars
import uuid

trace_var = contextvars.ContextVar("trace_id", default="SYSTEM")

class TraceFormatter(logging.Formatter):
    def format(self, record):
        # 如果第三方库（如 httpx）没带 trace_id，手动补上当前的
        if not hasattr(record, "trace_id"):
            record.trace_id = trace_var.get()
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(trace_id)s] - %(levelname)s - %(message)s"
)

# 劫持根 Logger 的 Handler，替换为我们的 Formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(TraceFormatter("%(asctime)s - [%(trace_id)s] - %(levelname)s - %(message)s"))

logger = logging.getLogger(__name__)

def set_trace(trace_id: str = None):
    if not trace_id:
        trace_id = uuid.uuid4().hex[:8]
    trace_var.set(trace_id)
    return trace_id

def get_trace():
    return trace_var.get()