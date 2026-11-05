import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.chat import router as chat_router
# ✅ 正确的新导入
from app.db.vector_store import init_knowledge_base
# 顶部导入
from app.db import models
from app.db.database import engine

# 在 app = FastAPI(...) 下面加上这句
models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="智能助手Agent服务")

# 👇 注册启动事件，确保子进程也能正确加载知识库
@app.on_event("startup")
def startup_event():
    init_knowledge_base(BASE_DIR)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.include_router(chat_router)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(STATIC_DIR):
    os.mkdir(STATIC_DIR)
    
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))