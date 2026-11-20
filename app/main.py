import os

# 1. 环境变量最先注入
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from slowapi.errors import RateLimitExceeded

from app.core.logger import logger
from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.api.health import router as health_router
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
    CORSMiddleware,
    # 带 Cookie 的跨域必须指明来源，不能用 *，否则浏览器会拒绝
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(health_router)

# 6.5 指标收集中间件（放在路由注册之后、静态挂载之前）
@app.middleware("http")
async def _metrics_middleware(request, call_next):
    import time as _time

    from app.core.metrics import record_http

    started = _time.perf_counter()
    response = await call_next(request)
    # /metrics 自己不计入，否则每次抓取都会污染统计
    if request.url.path != "/metrics":
        record_http(request.url.path, response.status_code, _time.perf_counter() - started)
    return response


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint():
    """Prometheus 抓取端点（文本暴露格式）。"""
    from fastapi.responses import PlainTextResponse

    from app.core.metrics import render_metrics

    return PlainTextResponse(render_metrics())


# 7. 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 8. 注册启动事件
_instance_lock = None


@app.on_event("startup")
def startup_event():
    # 单实例锁：两个进程同时打开同一个 SQLite 会互相锁死
    global _instance_lock
    from app.core.instance_lock import InstanceLock

    _instance_lock = InstanceLock(os.path.join(BASE_DIR, "data", "agent.lock"))
    if not _instance_lock.acquire():
        logger.error(
            "另一个实例正在使用同一个 data 目录（data/agent.lock 被占用）。\n"
            "两个进程同时打开 SQLite 会互相锁死，表现为 “database is locked”。\n"
            "请先停掉已有实例，或换一个工作目录。"
        )
        raise RuntimeError("检测到重复启动：请先停止已运行的实例")
    # 启动自检：配置类问题在这里说清楚，别等用户点发送才 500
    try:
        from app.core.selfcheck import validate_startup, has_fatal

        problems = validate_startup()
        for item in problems:
            level = item.get("level")
            message = f"[自检/{item.get('code')}] {item.get('message')}"
            if level == "fatal":
                logger.error(message)
            elif level == "warn":
                logger.warning(message)
            else:
                logger.info(message)
        if has_fatal(problems):
            logger.error("启动自检发现致命问题：服务能起来，但对话会失败，请按上面提示修复 .env")
    except Exception as e:
        logger.warning(f"启动自检异常（不影响服务启动）：{e}")

    from app.db.database import ensure_schema

    applied = ensure_schema()
    if applied:
        logger.info(f"[迁移] 已应用：{applied}")

    init_knowledge_base(BASE_DIR)


@app.on_event("shutdown")
def shutdown_event():
    global _instance_lock
    if _instance_lock:
        _instance_lock.release()
        _instance_lock = None

def _with_token(response):
    """给页面响应挂一个短期令牌 Cookie，前端代码里就不需要任何密钥。"""
    from app.middleware.auth import TOKEN_COOKIE, TOKEN_TTL, issue_token

    response.set_cookie(
        key=TOKEN_COOKIE,
        value=issue_token(),
        max_age=TOKEN_TTL,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response


@app.get("/")
def index():
    return _with_token(FileResponse(os.path.join(STATIC_DIR, "index.html")))


@app.post("/session/token", include_in_schema=False)
def session_token():
    """给命令行/脚本换一个短期令牌（浏览器走 Cookie，不需要调这个）。"""
    from app.middleware.auth import TOKEN_COOKIE, TOKEN_TTL, issue_token

    token = issue_token()
    response = JSONResponse({"code": 200, "data": {"token": token, "expires_in": TOKEN_TTL}})
    response.set_cookie(key=TOKEN_COOKIE, value=token, max_age=TOKEN_TTL,
                        httponly=True, samesite="strict", path="/")
    return response

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """用量仪表盘：token / 成本 / 延迟的独立页面。"""
    return _with_token(FileResponse(os.path.join(STATIC_DIR, "dashboard.html")))