import os
import json
from dotenv import load_dotenv

# 👇 核心修复：必须在读取任何环境变量之前，先加载 .env 文件！
load_dotenv()

from openai import OpenAI
from app.core.logger import logger

# ========== 大模型与服务配置 ==========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY") # 补上博查的 Key 读取
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 20))

# ========== 大模型调用超时（秒）==========
# 不设超时的话，网关一旦不回包，OpenAI SDK 默认要等 600s × 重试次数，
# 表现就是"整个服务卡死"（事件循环被 agent_app.invoke 占住）。
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# 删掉或替换原来的 client 初始化
try:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY,
                    timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
except Exception:
    client = None
# 默认模型：.env 里配 DEFAULT_MODEL_ID，没配就用 qwen-plus 兜底
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "qwen-plus")
MODEL_NAME = DEFAULT_MODEL_ID

# ========== 多模型注册表配置 ==========
MODEL_REGISTRY_STR = os.getenv("MODEL_REGISTRY", "{}")

def get_model_registry():
    """安全地解析多模型注册表 JSON"""
    try:
        return json.loads(MODEL_REGISTRY_STR)
    except Exception as e:
        logger.error(f"❌ 解析 MODEL_REGISTRY 失败，请检查 .env 格式：{e}")
        return {}


def get_model_extra_body(model_name: str) -> dict:
    """注册表里可以给单个模型追加请求参数，例如关掉深度思考：

        "doubao-seed-2-1-lite": {
            ...,
            "extra_body": {"thinking": {"type": "disabled"}}
        }

    豆包 2.1 Pro/Lite 默认开启深度思考，一次简单问答要 20 秒以上；
    关掉之后同样的请求 1.5 秒内返回。
    """
    registry = get_model_registry()
    config = registry.get(model_name) or {}
    if not config:
        # 传进来的可能是解析后的真实模型名（例如注册表 key 是 qwen-plus，
        # 实际发给厂商的是 qwen3.8-flash），按 model 字段反查一次
        for entry in registry.values():
            if entry.get("model") == model_name:
                config = entry
                break
    extra = config.get("extra_body")
    return extra if isinstance(extra, dict) else {}


def resolve_model_id(model_name: str) -> str:
    """注册表里可以用 model 字段覆盖真正发给厂商的模型名。

    比如火山方舟（豆包）要求 model 传推理接入点 ID（ep-xxxxxxxx），
    而注册表的 key 又想用个好读的名字，就可以这样配：
        "doubao-pro": {"model": "ep-20250101xxxxxx-xxxxx", ...}
    """
    registry = get_model_registry()
    config = registry.get(model_name) or {}
    return config.get("model") or model_name

# ========== RAG 检索配置 ==========
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 300))        # 文本切分大小
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 50))   # 文本切分重叠度
RAG_RECALL_TOP_K = int(os.getenv("RAG_RECALL_TOP_K", 10))     # 粗排召回数量
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", 1))        # 精排后最终返回数量

# ========== 文本分块配置 ==========
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "false").lower() == "true"  # 默认关闭
SEMANTIC_CHUNK_THRESHOLD = float(os.getenv("SEMANTIC_CHUNK_THRESHOLD", 0.85))  # 语义拆分阈值(0-1)

# ========== Agent 执行配置 ==========
MAX_TOOL_RETRY = int(os.getenv("MAX_TOOL_RETRY", 2))          # 工具最大错误重试次数
MAX_CONCURRENT_TOOLS = int(os.getenv("MAX_CONCURRENT_TOOLS", 5)) # 线程池最大并发数

# ========== 模型分层路由 ==========
# 开启后，前端传 auto_route=true 的请求会按任务复杂度自动挑模型档位；
# 关闭时行为与之前完全一致，不影响已有调用方。
AUTO_ROUTE = os.getenv("AUTO_ROUTE", "false").strip().lower() in ("1", "true", "yes", "on")
# 每次模型调用的用量记录（tokens / 成本 / 延迟），用于事后算账和对照实验
USAGE_LOG_PATH = os.getenv("USAGE_LOG_PATH", "data/usage.jsonl")