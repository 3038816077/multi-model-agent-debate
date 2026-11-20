"""SQLite 备份与恢复。

用法：
    python scripts/backup_db.py                 # 备份到 data/backups/
    python scripts/backup_db.py --keep 14       # 保留最近 14 份
    python scripts/backup_db.py --restore FILE  # 从备份恢复（会先备份当前库）

为什么用 sqlite3 的 backup API 而不是复制文件：
WAL 模式下直接 copy 数据库文件会漏掉还没落盘的 -wal 内容，恢复出来可能缺数据。
backup API 会保证拿到一致的快照。
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "agent.db")
BACKUP_DIR = os.path.join(REPO_ROOT, "data", "backups")


def backup(db_path: str = DB_PATH, backup_dir: str = BACKUP_DIR, keep: int = 7) -> str:
    """做一份一致性快照，返回备份文件路径。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库不存在：{db_path}")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(backup_dir, f"agent-{stamp}.db")

    source = sqlite3.connect(db_path, timeout=30)
    dest = sqlite3.connect(target)
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()

    prune(backup_dir, keep)
    return target


def prune(backup_dir: str = BACKUP_DIR, keep: int = 7) -> list:
    """只保留最近 keep 份，返回被删掉的文件。"""
    if not os.path.isdir(backup_dir):
        return []
    files = sorted(
        (os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.endswith(".db")),
        key=os.path.getmtime,
    )
    removed = files[:-keep] if keep > 0 else files
    for path in removed:
        try:
            os.remove(path)
        except OSError:
            pass
    return removed


def restore(backup_file: str, db_path: str = DB_PATH) -> str:
    """用备份覆盖当前库；覆盖前会先把当前库再备份一份。"""
    if not os.path.exists(backup_file):
        raise FileNotFoundError(f"备份文件不存在：{backup_file}")
    if os.path.exists(db_path):
        backup(db_path, BACKUP_DIR, keep=7)
    shutil.copy2(backup_file, db_path)
    # 旧的 WAL/SHM 与恢复后的库不匹配，必须清掉
    for suffix in ("-wal", "-shm"):
        stale = db_path + suffix
        if os.path.exists(stale):
            os.remove(stale)
    return db_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=7, help="保留最近几份备份")
    ap.add_argument("--restore", default="", help="从指定备份恢复")
    args = ap.parse_args()

    if args.restore:
        path = restore(args.restore)
        print(f"已恢复：{path}（原库已另存一份备份）")
        return 0

    path = backup(keep=args.keep)
    size = os.path.getsize(path)
    print(f"备份完成：{path}（{size / 1024:.0f} KB，保留最近 {args.keep} 份）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
