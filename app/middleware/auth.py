import os
import secrets
import time

from fastapi import Header, HTTPException, Request

# main.py 从这里导入限流异常处理器，重新导出（别删）
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: F401
from slowapi.errors import RateLimitExceeded  # noqa: F401
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# 鉴权：静态 API Key + 服务端下发的短期令牌
#
# 为什么不是「把 key 写进前端」：前端代码任何人都能看到，密钥一旦写死在
# HTML 里就等于公开。现在的做法是：
#   1. 打开页面时，服务端下发一个随机令牌（HttpOnly Cookie，SameSite=Strict）
#   2. 浏览器后续请求自动带上 Cookie，前端代码里没有任何密钥
#   3. 命令行/脚本仍然可以用 X-API-Key 或 Authorization: Bearer
# ---------------------------------------------------------------------------

TOKEN_COOKIE = "agent_token"
TOKEN_TTL = int(os.getenv("TOKEN_TTL", str(12 * 3600)))

# 进程内令牌表：单机单进程足够；多 worker 部署时改成 Redis 或签名令牌
_tokens: dict = {}


def issue_token() -> str:
    """发一个新令牌，并顺手清掉过期的。"""
    now = time.time()
    for key, exp in list(_tokens.items()):
        if exp < now:
            _tokens.pop(key, None)
    token = secrets.token_urlsafe(24)
    _tokens[token] = now + TOKEN_TTL
    return token


def token_valid(token: str) -> bool:
    if not token:
        return False
    exp = _tokens.get(token)
    if not exp:
        return False
    if exp < time.time():
        _tokens.pop(token, None)
        return False
    return True


def _get_limiter_key(request: Request):
    """限流粒度：优先按 API Key / 令牌独立计数，都没有就按 IP"""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    token = request.cookies.get(TOKEN_COOKIE)
    if token:
        return f"token:{token[:12]}"
    return f"ip:{get_remote_address(request)}"


# 评测/压测时可用 RATE_LIMIT_ENABLED=0 关闭限流，默认开启
limiter = Limiter(key_func=_get_limiter_key,
                  enabled=os.getenv("RATE_LIMIT_ENABLED", "1") != "0")


async def verify_api_key(request: Request, x_api_key: str = Header(None)):
    """鉴权依赖：静态 key、Bearer 令牌、浏览器 Cookie 三者满足其一即可。

    .env 里没配 API_KEY 时自动跳过校验（本地裸跑场景）。
    """
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return True
    if x_api_key and x_api_key == expected_key:
        return True
    authorization = request.headers.get("Authorization") or ""
    if authorization.lower().startswith("bearer "):
        if token_valid(authorization[7:].strip()):
            return True
    if token_valid(request.cookies.get(TOKEN_COOKIE, "")):
        return True
    raise HTTPException(status_code=401, detail="无效或过期的凭证")
