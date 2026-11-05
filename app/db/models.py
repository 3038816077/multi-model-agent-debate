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