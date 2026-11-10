import contextvars
from openai import OpenAI
from app.core.logger import logger
from app.core.config import get_model_registry, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_TIMEOUT, LLM_MAX_RETRIES

# 使用 contextvars 实现请求级别的上下文隔离
_llm_client_ctx = contextvars.ContextVar("llm_client", default=None)
_llm_model_name_ctx = contextvars.ContextVar("llm_model_name", default=None)

# 全局默认 Client (兜底)
try:
    _default_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
                             timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    _default_model_name = "qwen-plus"
except Exception as e:
    logger.error(f"初始化默认 LLM Client 失败: {e}")
    _default_client = None
    _default_model_name = None

_PLACEHOLDER_HINTS = ("填入", "你的", "xxx", "xxx")


def friendly_llm_error(exc: Exception, model_name: str = "") -> str:
    """把 SDK 的英文报错翻译成能直接照着改的中文提示。"""
    text = str(exc)
    low = text.lower()
    prefix = f"模型 {model_name} " if model_name else "模型调用"

    # 占位符 Key 里含中文时，httpx 在拼 Authorization 头会抛 ascii 编码错误，
    # 报错信息完全看不出真正原因，这里单独识别。
    if "ascii" in low and "encode" in low:
        return prefix + "的 API Key 含非 ASCII 字符（多半还是占位符），请到 .env 的 MODEL_REGISTRY 填真实 Key"
    if "403" in text or "quota" in low or "insufficient" in low or "balance" in low:
        return prefix + "调用被拒绝：额度不足或欠费（403），请到厂商控制台充值，或关闭「仅用免费额度」模式"
    if "401" in text or "invalid_api_key" in low or "unauthorized" in low or "authentication" in low:
        return prefix + "鉴权失败：API Key 无效（401），请检查 Key 是否过期或复制完整"
    if "404" in text or "model_not_found" in low or "does not exist" in low or "not found" in low:
        return prefix + "不存在：模型名写错了（404），检查 MODEL_REGISTRY 里的 key 和 model 字段"
    if "429" in text or "rate limit" in low or "too many requests" in low:
        return prefix + "触发限流（429），稍等几秒再试"
    if "503" in text or "no available accounts" in low:
        return prefix + "暂无可用账号（503）：中转站这个分组当前没有空闲账号，稍后重试，或换用别的模型分组"
    if "timeout" in low or "timed out" in low:
        return prefix + "请求超时，检查网络或调大 .env 里的 LLM_TIMEOUT"
    if "connection" in low or "getaddrinfo" in low or "ssl" in low:
        return prefix + "网络不通：请检查 base_url 是否正确、能否访问外网"
    return prefix + "调用失败：" + text[:200]


def mask_api_key(api_key: str) -> str:
    """日志脱敏"""
    if not api_key: return "None"
    if len(api_key) <= 8: return "***"
    return api_key[:4] + "****" + api_key[-4:]

# ==================== 修复：完全兼容 nodes.py 的元组传参 ====================
def get_llm_context():
    """获取当前上下文的快照（用于传递给子线程）"""
    return _llm_client_ctx.get(), _llm_model_name_ctx.get()

def set_llm_context(client, model_name):
    """在子线程中恢复上下文"""
    if client:
        _llm_client_ctx.set(client)
    if model_name:
        _llm_model_name_ctx.set(model_name)
# ==============================================================================

def set_llm_client(api_key: str = None, base_url: str = None, model_name: str = None):
    """
    动态设置当前请求的 LLM 客户端
    """
    registry = get_model_registry()

    # 情况一：前端选了内置模型（走系统注册表）
    if model_name and not api_key:
        if model_name in registry:
            config = registry[model_name]
            api_key = config.get("api_key")
            base_url = config.get("base_url")
            # 注册表 key 只用于前端展示；真正发给厂商的模型名允许用 model 字段覆盖
            model_name = config.get("model") or model_name
        else:
            logger.warning(f"⚠️ 请求的模型 {model_name} 不在注册表中，将回退到默认模型。")
            _llm_client_ctx.set(_default_client)
            _llm_model_name_ctx.set(_default_model_name)
            return

    # 情况二：前端传了自定义 Key（走右上角透传）
    if api_key:
        if not base_url:
            if model_name and model_name in registry:
                base_url = registry[model_name].get("base_url")
            else:
                base_url = OPENAI_BASE_URL
        if not model_name:
            model_name = "custom-model"
            
        try:
            custom_client = OpenAI(api_key=api_key, base_url=base_url,
                                   timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
            _llm_client_ctx.set(custom_client)
            _llm_model_name_ctx.set(model_name)
            logger.info(f"✅ 切换至模型: {model_name} ({base_url})")
        except Exception as e:
            logger.error(f"❌ 设置自定义 Client 失败: {e}")
            _llm_client_ctx.set(_default_client)
            _llm_model_name_ctx.set(_default_model_name)
            return

    # 情况三：什么都没传（完全兜底）
    if not api_key and not model_name:
        _llm_client_ctx.set(_default_client)
        _llm_model_name_ctx.set(_default_model_name)


def get_llm_client() -> OpenAI:
    """获取当前请求的 LLM Client"""
    client = _llm_client_ctx.get()
    if client is None:
        return _default_client
    return client

def get_model_name() -> str:
    """获取当前请求使用的模型名"""
    name = _llm_model_name_ctx.get()
    if name is None:
        return _default_model_name
    return name