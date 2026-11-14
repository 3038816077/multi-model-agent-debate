# -*- coding: utf-8 -*-
"""查看 / 清空跨会话记忆。

用法（在项目根目录执行）：
    python scripts/show_memory.py          # 查看
    python scripts/show_memory.py --clear  # 清空
"""
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "agent.db")


def main():
    if not os.path.exists(DB):
        print("找不到数据库：%s（先启动一次服务，会自动建表）" % DB)
        return

    conn = sqlite3.connect(DB)
    try:
        if "--clear" in sys.argv:
            conn.execute("update user_memory set content='' where id='global'")
            conn.commit()
            print("已清空跨会话记忆")
            return

        row = conn.execute(
            "select content, updated_at from user_memory where id='global'"
        ).fetchone()
    finally:
        conn.close()

    if not row or not row[0]:
        print("（记忆为空）")
        return

    print("=== 当前跨会话记忆 ===")
    print(row[0])
    print()
    print("更新时间：%s" % row[1])


if __name__ == "__main__":
    main()
