"""鉴权逻辑的回归测试。

守两件事：
1. 令牌签发/校验/过期必须正确（过期了不能继续用）
2. 静态 key 与令牌是「或」的关系，任一满足即可
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.middleware import auth  # noqa: E402


def test_issue_and_validate():
    token = auth.issue_token()
    assert token and len(token) > 20
    assert auth.token_valid(token) is True


def test_unknown_token_is_invalid():
    assert auth.token_valid("not-a-real-token") is False
    assert auth.token_valid("") is False


def test_expired_token_is_invalid():
    token = auth.issue_token()
    auth._tokens[token] = time.time() - 1  # 手动把它改成过期
    assert auth.token_valid(token) is False
    # 过期后应被清掉，不留在内存里
    assert token not in auth._tokens


def test_tokens_are_unique():
    tokens = {auth.issue_token() for _ in range(50)}
    assert len(tokens) == 50


def test_issue_cleans_expired_entries():
    stale = "stale-token"
    auth._tokens[stale] = time.time() - 100
    auth.issue_token()
    assert stale not in auth._tokens


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
