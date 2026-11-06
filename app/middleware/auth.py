import os
from fastapi import Header, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

async def verify_api_key(x_api_key: str = Header(None)):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return True # 没配就不拦截，方便本地测试
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return True