import os
from dotenv import load_dotenv

# 👇 核心修复：必须在读取任何环境变量之前，先加载 .env 文件！
load_dotenv()

from openai import OpenAI
from app.core.logger import logger

# ========== 大模型与服务配置 ==========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 20))
# 删掉或替换原来的 client 初始化
try:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
except Exception:
    client = None
MODEL_NAME = "qwen-plus"

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