"""健康检查：给部署和排查用。

/health        轻量：进程活着 + 注册表可用 + 数据目录可写（不调用模型，秒回）
/health/deep   逐个模型发一次最小请求，确认 key 真的有效（慢，会花钱，慎用）
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

import os

from app.core.config import get_model_registry
from app.core.logger import logger
from app.core.selfcheck import check_registry, check_tiers, check_prices, check_writable, has_fatal
from app.core.router import MODEL_PRICE_PER_M, MODEL_TIERS

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _collect():
    registry = get_model_registry()
    problems = []
    problems += check_registry(registry)
    problems += check_tiers(registry, MODEL_TIERS)
    problems += check_prices(MODEL_PRICE_PER_M, registry)
    problems += check_writable([
        os.path.join(BASE_DIR, "data"),
        os.path.join(BASE_DIR, "data", "uploads"),
    ])
    return registry, problems


def health_payload() -> tuple:
    """返回 (HTTP 状态码, body)。抽成函数是为了能单测，不必起服务。"""
    registry, problems = _collect()
    fatal = has_fatal(problems)
    body = {
        "status": "error" if fatal else ("degraded" if problems else "ok"),
        "models": len(registry or {}),
        "tiers": {k: list(v) for k, v in (MODEL_TIERS or {}).items()},
        "problems": problems,
    }
    return (503 if fatal else 200), body


@router.get("/health", summary="健康检查（不调用模型）")
def health():
    code, body = health_payload()
    return JSONResponse(status_code=code, content=body)


@router.get("/health/deep", summary="深度检查：逐个模型发最小请求")
def health_deep():
    from app.api.chat import models_health  # 复用已有的逐模型探测

    return models_health()
