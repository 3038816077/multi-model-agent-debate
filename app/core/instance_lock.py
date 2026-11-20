"""单实例锁：同一个 data 目录只允许一个服务进程。

背景：SQLite 被两个进程同时打开时，即使开了 WAL 也会互相锁住，
表现为「database is locked」，而且排查起来很隐蔽（两个终端各跑一个）。

做法：在锁文件上拿操作系统的排他锁——进程一退出锁自动释放，
不会留下需要手工清理的僵尸锁（比「写 PID 再判断进程存活」可靠）。
"""
import os


class InstanceLock:
    def __init__(self, path: str):
        self.path = path
        self._handle = None

    def acquire(self) -> bool:
        """非阻塞加锁；成功返回 True，已被占用返回 False。"""
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handle = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt

                # 必须以同一个偏移量加锁/解锁，否则解锁会失败
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        # 记下 PID，方便排查到底是哪个进程占着
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
        except OSError:
            pass
        self._handle = handle
        return True

    def release(self) -> None:
        if not self._handle:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None
