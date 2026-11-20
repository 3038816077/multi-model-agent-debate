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
    connect_args={"check_same_thread": False, "timeout": 30}
)

# 👇 核心：监听每次建立连接的事件，强制执行 WAL 模式等 PRAGMA 命令
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # 开启 WAL 模式，允许读写并发（这是你在简历上写的硬核点）
    cursor.execute("PRAGMA journal_mode=WAL;")
    # 将同步模式设为 NORMAL，在保证数据安全的前提下提升写入性能
    cursor.execute("PRAGMA synchronous=NORMAL;")
    # 多个连接抢写时最多排队 15 秒再报错，而不是立刻 "database is locked"
    cursor.execute("PRAGMA busy_timeout=15000;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def ensure_schema() -> list:
    """轻量迁移：给已存在的库补上新增列。

    项目没有引入 Alembic（单机 SQLite，迁移脚本比依赖更省心），
    但也不能指望用户删库重来，所以用 PRAGMA 检查后 ALTER TABLE。
    返回本次实际执行的 SQL，便于启动日志里看到发生了什么。
    """
    from sqlalchemy import inspect, text

    applied = []
    inspector = inspect(engine)
    if "sessions" not in inspector.get_table_names():
        Base.metadata.create_all(bind=engine)
        return ["create_all"]

    columns = {c["name"] for c in inspector.get_columns("sessions")}
    if "archived" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN archived INTEGER DEFAULT 0"))
        applied.append("ALTER TABLE sessions ADD COLUMN archived")
    return applied
