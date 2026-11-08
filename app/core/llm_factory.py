import os
import contextvars
from openai import OpenAI
from app.core.config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME

ALLOWED_BASE_URLS = [
    "https://maas.qianwenaiapi.com",
    "https://dashscope.aliyuncs.com",
    "https://api.deepseek.com",
    "https://open.bigmodel.cn",
    "https://api.moonshot.cn",
    "https://api.openai.com",
    "https://api.siliconflow.cn",
    "https://ark.cn-beijing.volces.com"
]

_llm_client_ctx = contextvars.ContextVar("llm_client", default=None)
_model_name_ctx = contextvars.ContextVar("model_name", default=None)

def mask_api_key(key: str) -> str:
    if not key or len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"

def set_llm_client(user_api_key: str = None, user_base_url: str = None, user_model_name: str = None):
    if not user_api_key:
        _llm_client_ctx.set(None)
        _model_name_ctx.set(None)
        return

    use_url = user_base_url or OPENAI_BASE_URL
    if not any(use_url.startswith(allowed) for allowed in ALLOWED_BASE_URLS):
        raise ValueError(f"不支持的模型接口地址: {use_url}")

    new_client = OpenAI(base_url=use_url, api_key=user_api_key)
    _llm_client_ctx.set(new_client)
    _model_name_ctx.set(user_model_name)

def get_llm_client() -> OpenAI:
    client = _llm_client_ctx.get()
    if client:
        return client
    return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

def get_model_name() -> str:
    name = _model_name_ctx.get()
    return name if name else MODEL_NAME

# 为了子线程透传
def get_llm_context():
    return _llm_client_ctx.get(), _model_name_ctx.get()

def set_llm_context(client, model_name):
    if client:
        _llm_client_ctx.set(client)
    if model_name:
        _model_name_ctx.set(model_name)