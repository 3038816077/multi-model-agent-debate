import os
from fastapi import Header, HTTPException, Request, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

def _get_limiter_key(request: Request):
    """限流粒度策略：优先按 API Key 独立计数，无 Key 回退到 IP"""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    return f"ip:{get_remote_address(request)}"

# 限流器初始化，使用自定义键策略
limiter = Limiter(key_func=_get_limiter_key)

async def verify_api_key(x_api_key: str = Header(None)):
    """API Key 鉴权依赖，未配置时自动跳过校验"""
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return True
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return True