import os

# 1. 环境变量最先注入
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi.errors import RateLimitExceeded

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.db.vector_store import init_knowledge_base
from app.db import models
from app.db.database import engine
from app.middleware.auth import limiter, _rate_limit_exceeded_handler

# 2. 初始化数据库表
models.Base.metadata.create_all(bind=engine)

# 3. 定义路径（必须放在 startup_event 之前，否则会报错）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.mkdir(STATIC_DIR)

# 4. 初始化 FastAPI
app = FastAPI(title="多模型研讨 Agent 服务")

# 5. 注册限流器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 6. 注册中间件与路由
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.include_router(chat_router)
app.include_router(upload_router)

# 7. 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 8. 注册启动事件
@app.on_event("startup")
def startup_event():
    init_knowledge_base(BASE_DIR)

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))