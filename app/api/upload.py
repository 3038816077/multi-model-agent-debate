"""文档上传与解析接口。

前端把 Word / PDF 读成 base64 传上来（这样不用引入 python-multipart 依赖），
后端解析成纯文本后存到 data/uploads/<doc_id>.json，
后续对话只要带上 attachment_id，就会把文档内容注入到模型上下文里。
"""
import base64
import json
import os
import re
import uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.core.logger import logger
from app.middleware.auth import limiter, verify_api_key
from app.tools.doc_reader import DocReadError, SUPPORTED_EXTS, extract_text_from_bytes

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_BYTES = 15 * 1024 * 1024      # 单文件 15MB
PREVIEW_CHARS = 200               # 返回给前端展示的开头预览
_DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class UploadRequest(BaseModel):
    filename: str
    content_base64: str


def _paths(doc_id: str):
    return os.path.join(UPLOAD_DIR, doc_id + ".json")


def load_document(doc_id: str):
    """按 doc_id 取回解析结果，取不到返回 None（例如服务重启后文件被清了）。"""
    if not doc_id or not _DOC_ID_RE.match(doc_id):
        return None
    path = _paths(doc_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"读取文档 {doc_id} 失败: {e}")
        return None


@router.post("/upload", summary="上传 Word / PDF 并解析为纯文本")
@limiter.limit("20/minute")
async def upload_document(request: Request, body: UploadRequest, auth: bool = Depends(verify_api_key)):
    filename = os.path.basename((body.filename or "").strip()) or "未命名文件"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        return {"code": 400, "message": f"暂不支持 {ext or '该'} 格式，目前支持：{'、'.join(sorted(SUPPORTED_EXTS))}"}

    # base64 可能带 data URL 前缀（data:application/pdf;base64,xxxx）
    raw_b64 = body.content_base64 or ""
    if raw_b64.startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[-1]
    try:
        data = base64.b64decode(raw_b64, validate=False)
    except Exception:
        return {"code": 400, "message": "文件内容解码失败，请重新上传"}

    if not data:
        return {"code": 400, "message": "文件是空的"}
    if len(data) > MAX_BYTES:
        return {"code": 413, "message": f"文件太大了（{len(data) / 1024 / 1024:.1f}MB），上限 15MB"}

    try:
        parsed = await __import__("asyncio").to_thread(extract_text_from_bytes, data, filename)
    except DocReadError as e:
        return {"code": 422, "message": str(e)}
    except Exception as e:
        logger.error(f"文档解析异常: {e}")
        return {"code": 500, "message": f"文档解析失败：{e}"}

    doc_id = uuid.uuid4().hex
    record = {
        "doc_id": doc_id,
        "filename": filename,
        "kind": parsed["kind"],
        "chars": parsed["chars"],
        "warning": parsed["warning"],
        "text": parsed["text"],
        "size": len(data),
    }
    try:
        with open(_paths(doc_id), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存文档失败: {e}")
        return {"code": 500, "message": "文档解析成功但保存失败，请检查磁盘权限"}

    logger.info(f"📄 文档上传成功: {filename} ({parsed['kind']}, {parsed['chars']} 字) -> {doc_id}")
    return {
        "code": 200,
        "data": {
            "doc_id": doc_id,
            "filename": filename,
            "kind": parsed["kind"],
            "chars": parsed["chars"],
            "warning": parsed["warning"],
            "preview": parsed["text"][:PREVIEW_CHARS],
        },
    }
