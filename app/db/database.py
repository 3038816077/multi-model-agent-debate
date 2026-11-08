import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

# 👇 新增：自动检查并创建 data 文件夹
if not os.path.exists("./data"):
    os.makedirs("./data")

# SQLite 数据库文件会生成在 data 文件夹下
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/agent.db"

# check_same_thread=False 是为了解决 FastAPI 多线程访问 SQLite 的报错
# 👇 timeout=15 是为了在并发极高、发生锁等待时，给予足够的排队时间，避免直接报错
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False, "timeout": 15}
)

# 👇 核心：监听每次建立连接的事件，强制执行 WAL 模式等 PRAGMA 命令
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # 开启 WAL 模式，允许读写并发（这是你在简历上写的硬核点）
    cursor.execute("PRAGMA journal_mode=WAL;")
    # 将同步模式设为 NORMAL，在保证数据安全的前提下提升写入性能
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()