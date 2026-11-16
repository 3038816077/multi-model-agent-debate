"""图片识别：把用户上传的图片交给视觉模型，转成可进上下文的文字。

设计取舍：不做"每个模型都看图"，而是先识别成文字再走原有流程。
理由是并行模式下不同模型的视觉能力参差不齐（实测 glm-4-flash 直接拒收图片），
先统一转文字，后面谁都能用同一份材料。
"""
import base64
import mimetypes
import os

from app.core.config import (
    client as default_client,
    get_model_max_tokens,
    get_model_registry,
    resolve_model_id,
)
from app.core.logger import logger
from app.core.router import log_usage
from app.tools.doc_reader import DocReadError

# 显式标记优先，否则按这个顺序试；都不可用就报错
FALLBACK_VISION_MODELS = (
    "qwen-plus",
    "qwen3.8-omni-flash",
    "gpt-5.6",
    "doubao-seed-2-1-lite",
    "kimi-k3.1-pro",
)

VISION_PROMPT = """把这张图片的内容转成文字，供后续提问使用：

1. 完整抄录图中所有文字，保持原有结构（表格用 | 分隔，列表保留编号）
2. 如果是图表 / 截图 / 照片，再用两三句话说明它展示的核心信息和关键数字
3. 不要评价、不要总结成一句话，尽量保留细节；看不清的地方写"（不清）"

只输出内容本身，不要加"以下是图片内容"这类开场白。"""


def pick_vision_model() -> str:
    """挑一个能看图的模型：注册表里标了 vision: true 的优先。"""
    registry = get_model_registry()
    for key, cfg in registry.items():
        if isinstance(cfg, dict) and cfg.get("vision") is True:
            return key
    for key in FALLBACK_VISION_MODELS:
        if key in registry:
            return key
    return ""


def describe_image(data: bytes, filename: str) -> dict:
    """识别图片内容，返回和 doc_reader 一致的字典结构。"""
    model = pick_vision_model()
    if not model:
        raise DocReadError("没有可用的视觉模型，请在 .env 的注册表里给某个模型加 \"vision\": true")

    mime = mimetypes.guess_type(filename)[0] or "image/png"
    if mime == "image/svg+xml":
        raise DocReadError("暂不支持 SVG，请导出为 PNG 或 JPG 再上传")
    payload = "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())

    try:
        c = default_client
        resp = c.chat.completions.create(
            model=resolve_model_id(model),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": payload}},
                ],
            }],
            max_tokens=min(get_model_max_tokens(model, 1500), 4000),
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            u = getattr(resp, "usage", None)
            log_usage(trace_id="", model=model, scene="vision",
                      prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                      completion_tokens=getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pass
    except Exception as e:
        logger.error("图片识别失败（%s）：%s" % (model, e))
        raise DocReadError("图片识别失败：%s" % str(e)[:120])

    if not text:
        raise DocReadError("图片里没有识别出可用内容，换一张更清晰的图试试")

    return {"text": text, "chars": len(text), "kind": "图片", "warning": "", "model": model}
