import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.models import DBSession, DBMessage

def get_messages(db: Session, session_id: str, limit: int = 20) -> list:
    """从数据库拉取最近 N 条历史消息"""
    # 按时间倒序取 limit 条，然后再反转回正序
    msgs = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.created_at.desc()).limit(limit).all()
    msgs.reverse()
    
    # 确保第一条永远是 system 提示词（防止被 limit 截掉）
    if not msgs or msgs[0].role != "system":
        system_msg = db.query(DBMessage).filter(DBMessage.session_id == session_id, DBMessage.role == "system").first()
        if system_msg:
            msgs.insert(0, system_msg)
            
    return [{"role": m.role, "content": m.content} for m in msgs]
def save_message(db: Session, session_id: str, role: str, content: str, thought_process: str = ""):
    """保存单条消息。

    顺便保证 sessions 表里有这条会话的记录：/sessions 列表是查 sessions 表的，
    以前只有 update_long_term_memory 才会建这条记录，所以多模型研讨聊完
    左侧历史里根本看不到（用户反馈的“没加入普通会话”就是这个原因）。
    """
    sess = get_or_create_session(db, session_id)
    sess.updated_at = datetime.now()   # 让会话按最后活跃时间排序
    msg = DBMessage(
        id=str(uuid.uuid4()), 
        session_id=session_id, 
        role=role, 
        content=content,
        thought_process=thought_process
    )
    db.add(msg)
    db.commit()

def get_or_create_session(db: Session, session_id: str) -> DBSession:
    """获取或创建会话（用于长期记忆）"""
    db_sess = db.query(DBSession).filter(DBSession.session_id == session_id).first()
    if not db_sess:
        db_sess = DBSession(session_id=session_id, long_term_memory="")
        db.add(db_sess)
        db.commit()
        db.refresh(db_sess)
    return db_sess

def update_long_term_memory(db: Session, session_id: str, memory: str):
    """更新长期记忆"""
    db_sess = get_or_create_session(db, session_id)
    db_sess.long_term_memory = memory
    db.commit()