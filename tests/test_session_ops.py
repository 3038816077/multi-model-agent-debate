"""会话批量操作（列表 / 归档 / 删除）的单元测试。

用内存 SQLite，不碰真实的 data/agent.db。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import DBMessage, DBSession
from app.db.session_ops import (
    _clean_ids,
    delete_sessions,
    list_sessions,
    set_archived,
)


def _db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _seed(db, sid, text="问题", archived=0):
    db.add(DBSession(session_id=sid, archived=archived))
    db.add(DBMessage(id="m-" + sid, session_id=sid, role="user", content=text))
    db.commit()


def test_list_splits_active_and_archived():
    db = _db()
    _seed(db, "a")
    _seed(db, "b", archived=1)
    data = list_sessions(db, include_archived=True)
    assert [x["session_id"] for x in data["active"]] == ["a"]
    assert [x["session_id"] for x in data["archived"]] == ["b"]
    assert data["active"][0]["title"] == "问题"


def test_list_can_exclude_archived():
    db = _db()
    _seed(db, "a")
    _seed(db, "b", archived=1)
    data = list_sessions(db, include_archived=False)
    assert len(data["active"]) == 1 and data["archived"] == []


def test_archive_and_unarchive():
    db = _db()
    _seed(db, "a")
    _seed(db, "b")
    assert set_archived(db, ["a", "b"], True)["updated"] == 2
    data = list_sessions(db, include_archived=True)
    assert len(data["active"]) == 0 and len(data["archived"]) == 2
    assert set_archived(db, ["a"], False)["updated"] == 1
    data = list_sessions(db, include_archived=True)
    assert [x["session_id"] for x in data["active"]] == ["a"]


def test_delete_removes_session_and_messages():
    db = _db()
    _seed(db, "a")
    _seed(db, "b")
    result = delete_sessions(db, ["a"])
    assert result["deleted_sessions"] == 1
    assert result["deleted_messages"] == 1
    assert [x["session_id"] for x in list_sessions(db)["active"]] == ["b"]
    assert db.query(DBMessage).filter(DBMessage.session_id == "a").count() == 0


def test_delete_ignores_unknown_ids():
    db = _db()
    _seed(db, "a")
    result = delete_sessions(db, ["不存在", "a"])
    assert result["deleted_sessions"] == 1
    assert result["ids"] == ["a"]


def test_delete_empty_input_is_noop():
    db = _db()
    _seed(db, "a")
    assert delete_sessions(db, [])["deleted_sessions"] == 0
    assert delete_sessions(db, None)["deleted_sessions"] == 0


def test_clean_ids_dedupes_and_caps():
    assert _clean_ids(["a", "a", " b ", ""]) == ["a", "b"]
    assert len(_clean_ids([str(i) for i in range(500)])) == 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
