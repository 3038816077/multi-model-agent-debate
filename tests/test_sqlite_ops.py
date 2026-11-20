"""SQLite 运维相关测试：PRAGMA、单实例锁、备份。

这些是「数据库被并发写锁死」之后补的东西，守的是运行期行为，不是业务逻辑。
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from app.core.instance_lock import InstanceLock  # noqa: E402
from app.db.database import engine  # noqa: E402
import backup_db  # noqa: E402


def test_wal_and_busy_timeout_are_applied():
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() >= 15000
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        # 0=OFF 1=NORMAL 2=FULL
        assert conn.exec_driver_sql("PRAGMA synchronous").scalar() == 1


def test_lock_excludes_other_process():
    """另一个进程持锁时，本进程必须拿不到。"""
    d = tempfile.mkdtemp()
    lock_path = os.path.join(d, "agent.lock")
    holder_path = os.path.join(d, "holder.py")
    with open(holder_path, "w", encoding="utf-8") as f:
        f.write(
            "import sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from app.core.instance_lock import InstanceLock\n"
            "lock = InstanceLock(sys.argv[1])\n"
            "print(lock.acquire(), flush=True)\n"
            "time.sleep(2)\n" % ROOT
        )
    child = subprocess.Popen(
        [sys.executable, holder_path, lock_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = child.stdout.readline().strip()
        err = child.stderr.read() if not line else ""
        assert line == "True", f"子进程没拿到锁: {line!r} {err[-300:]}"
        mine = InstanceLock(lock_path)
        assert mine.acquire() is False
        mine.release()
    finally:
        child.wait(timeout=15)
    # 子进程退出后锁自动释放，不会留下僵尸锁
    after = InstanceLock(lock_path)
    assert after.acquire() is True
    after.release()


def test_backup_produces_readable_copy():
    d = tempfile.mkdtemp()
    db = os.path.join(d, "agent.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (v) VALUES ('hello')")
    conn.commit()
    conn.close()

    out = backup_db.backup(db_path=db, backup_dir=os.path.join(d, "backups"), keep=3)
    copy = sqlite3.connect(out)
    try:
        assert copy.execute("SELECT v FROM t").fetchone()[0] == "hello"
    finally:
        copy.close()


def test_prune_keeps_newest():
    d = tempfile.mkdtemp()
    for i in range(5):
        path = os.path.join(d, "agent-%d.db" % i)
        open(path, "w").close()
        os.utime(path, (time.time() + i, time.time() + i))
    removed = backup_db.prune(d, keep=2)
    assert len(removed) == 3
    assert len([f for f in os.listdir(d) if f.endswith(".db")]) == 2
    # 保留的必须是最新的两份
    left = sorted(os.listdir(d))
    assert "agent-3.db" in left and "agent-4.db" in left


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
