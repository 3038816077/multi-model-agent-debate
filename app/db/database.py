from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite 数据库文件会生成在项目根目录下
SQLALCHEMY_DATABASE_URL = "sqlite:///./agent.db"

# check_same_thread=False 是为了解决 FastAPI 多线程访问 SQLite 的报错
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()