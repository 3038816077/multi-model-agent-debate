import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 👇 新增：自动检查并创建 data 文件夹
if not os.path.exists("./data"):
    os.makedirs("./data")

# SQLite 数据库文件会生成在 data 文件夹下
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/agent.db"

# check_same_thread=False 是为了解决 FastAPI 多线程访问 SQLite 的报错
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()