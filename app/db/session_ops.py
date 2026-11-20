"""会话的批量操作：列表、归档、永久删除。

放在 db 层而不是路由里，是为了能脱离 FastAPI 单测（见 tests/test_session_ops.py）。
"""
from sqlalchemy.orm import Session

from app.db.models import DBMessage, DBSession

MAX_BATCH = 200


def _titles(db: Session, session_ids: list) -> dict:
    """每个会话的第一句用户提问，用作列表标题。"""
    if not session_ids:
        return {}
    rows = (
        db.query(DBMessage)
        .filter(DBMessage.session_id.in_(session_ids), DBMessage.role == "user")
        .order_by(DBMessage.created_at.asc())
        .all()
    )
    titles = {}
    for row in rows:
        if row.session_id in titles:
            continue
        text = (row.content or "").strip().replace("\n", " ")
        if text:
            titles[row.session_id] = text[:15] + ("..." if len(text) > 15 else "")
    return titles


def list_sessions(db: Session, include_archived: bool = False) -> dict:
    """返回 {active: [...], archived: [...]}，各自按更新时间倒序。"""
    query = db.query(DBSession)
    if not include_archived:
        query = query.filter((DBSession.archived == 0) | (DBSession.archived.is_(None)))
    sessions = query.order_by(DBSession.updated_at.desc()).all()

    titles = _titles(db, [s.session_id for s in sessions])
    active, archived = [], []
    for s in sessions:
        item = {
            "session_id": s.session_id,
            "title": titles.get(s.session_id, "新会话"),
            "updated_at": s.updated_at.strftime("%m-%d %H:%M") if s.updated_at else "",
            "archived": bool(s.archived or 0),
        }
        (archived if item["archived"] else active).append(item)
    return {"active": active, "archived": archived}


def _clean_ids(session_ids) -> list:
    ids = [str(x).strip() for x in (session_ids or []) if str(x).strip()]
    # 去重且保序
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:MAX_BATCH]


def delete_sessions(db: Session, session_ids) -> dict:
    """永久删除会话及其全部消息，返回实际删除数量。"""
    ids = _clean_ids(session_ids)
    if not ids:
        return {"deleted_sessions": 0, "deleted_messages": 0, "ids": []}

    existing = [
        row[0]
        for row in db.query(DBSession.session_id).filter(DBSession.session_id.in_(ids)).all()
    ]
    if not existing:
        return {"deleted_sessions": 0, "deleted_messages": 0, "ids": []}

    deleted_messages = (
        db.query(DBMessage).filter(DBMessage.session_id.in_(existing)).delete(synchronize_session=False)
    )
    deleted_sessions = (
        db.query(DBSession).filter(DBSession.session_id.in_(existing)).delete(synchronize_session=False)
    )
    db.commit()
    return {
        "deleted_sessions": int(deleted_sessions or 0),
        "deleted_messages": int(deleted_messages or 0),
        "ids": existing,
    }


def set_archived(db: Session, session_ids, archived: bool = True) -> dict:
    """归档 / 取消归档。"""
    ids = _clean_ids(session_ids)
    if not ids:
        return {"updated": 0, "ids": []}
    updated = (
        db.query(DBSession)
        .filter(DBSession.session_id.in_(ids))
        .update({DBSession.archived: 1 if archived else 0}, synchronize_session=False)
    )
    db.commit()
    return {"updated": int(updated or 0), "ids": ids}
