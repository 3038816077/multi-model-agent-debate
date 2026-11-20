"""pytest 入口：先装替身，再让测试导入应用模块。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _stubs import install  # noqa: E402

install()
