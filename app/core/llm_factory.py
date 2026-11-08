import os
import contextvars
from openai import OpenAI
from app.core.config import OPENAI_BASE_URL, OPENAI_API_KEY

# 允许的自定义接口白名单，防止被利用做 SSRF 攻击
ALLOWED_BASE_URLS = [
    "https://maas.qianwenaiapi.com",
    "https://dashscope.aliyuncs.com",
    "https://api.deepseek.com",
    "https://open.bigmodel.cn",
    "https://api.moonshot.cn",
    "https://api.openai.com"
]

# 定义请求级别的上下文
_llm_client_ctx = contextvars.ContextVar("llm_client", default=None)

def mask_api_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"

def set_llm_client(user_api_key: str = None, user_base_url: str = None):
    """根据用户传入的 Key 动态创建 Client"""
    if not user_api_key:
        _llm_client_ctx.set(None) # 恢复默认
        return

    use_url = user_base_url or OPENAI_BASE_URL
    # 白名单校验
    if not any(use_url.startswith(allowed) for allowed in ALLOWED_BASE_URLS):
        raise ValueError("不支持的模型接口地址")

    new_client = OpenAI(base_url=use_url, api_key=user_api_key)
    _llm_client_ctx.set(new_client)

def get_llm_client() -> OpenAI:
    """获取当前请求的 Client"""
    client = _llm_client_ctx.get()
    if client:
        return client
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
def get_llm_context():
    return _llm_client_ctx.get()

def set_llm_context(client):
    _llm_client_ctx.set(client)