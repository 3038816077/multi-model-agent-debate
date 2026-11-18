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
# 默认模型：.env 里配 DEFAULT_MODEL_ID，没配就用 doubao-seed-2-1 兜底
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "doubao-seed-2-1")
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


def _lookup_model_config(model_name: str) -> dict:
    """按注册表 key 找配置；找不到就按 model 字段反查（传进来的可能是真实模型名）。"""
    registry = get_model_registry()
    config = registry.get(model_name) or {}
    if not config:
        for entry in registry.values():
            if entry.get("model") == model_name:
                return entry
    return config


def get_model_max_tokens(model_name: str, default: int = 2048) -> int:
    """注册表可以给单个模型指定输出上限。

    思考型模型（DeepSeek V4.1 Flash、Kimi K3.1 Pro 等）会先把大量输出 token
    花在推理上，2048 很容易被思考吃光，正文返回空字符串——HTTP 200，
    但用户看到一个空气泡。给这类模型调大即可。
    """
    config = _lookup_model_config(model_name)
    try:
        value = int(config.get("max_tokens") or 0)
    except (TypeError, ValueError):
        value = 0
    return value or default


def get_model_extra_body(model_name: str) -> dict:
    """注册表里可以给单个模型追加请求参数，例如关掉深度思考：

        "doubao-seed-2-1-lite": {
            ...,
            "extra_body": {"thinking": {"type": "disabled"}}
        }

    豆包 2.1 Pro/Lite 默认开启深度思考，一次简单问答要 20 秒以上；
    关掉之后同样的请求 1.5 秒内返回。
    """
    config = _lookup_model_config(model_name)
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
# 评测/对照实验时设为 0：不读写跨会话记忆，避免并发请求互相污染上下文
ENABLE_USER_MEMORY = os.getenv("ENABLE_USER_MEMORY", "1") != "0"

# 关掉时间/算术的确定性快速通道。对照实验用：否则单模型在这些题上不调模型就算对
ENABLE_FAST_REPLY = os.getenv("ENABLE_FAST_REPLY", "1") != "0"

# 分歧分析用哪个模型：chief = 调用方指定的主模型（质量优先，慢一点）
#                    cheap = 便宜档（快，但分析更浅）
DIVERGENCE_MODEL = os.getenv("DIVERGENCE_MODEL", "chief").strip().lower()

# 分歧复核：每条分歧再让一个模型判一次「这两句真的矛盾吗」，不通过就丢弃。
# 留空 = 不复核；填注册表里的模型 key 即可开启。复核失败的条目会保留，避免误杀。
DIVERGENCE_VERIFIER = os.getenv("DIVERGENCE_VERIFIER", "").strip()

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