from sqlalchemy import Column, String, Text, DateTime
from datetime import datetime
from app.db.database import Base

class DBSession(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    long_term_memory = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class DBMessage(Base):
    __tablename__ = "messages"
    
    id = Column(String, primary_key=True, index=True) # 用 uuid 生成
    session_id = Column(String, index=True)
    role = Column(String) # system, user, assistant, tool
    content = Column(Text)
    thought_process = Column(Text, default="") # 顺便把思考过程也存下来
    created_at = Column(DateTime, default=datetime.now)


class DBUserMemory(Base):
    """跨会话的用户级长期记忆（单用户应用，固定一行 id="global"）。

    和 DBSession.long_term_memory 的区别：
    那个只对当前会话生效，换会话就没了；这张表全局共享。
    """
    __tablename__ = "user_memory"

    id = Column(String, primary_key=True)
    content = Column(Text, default="")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
