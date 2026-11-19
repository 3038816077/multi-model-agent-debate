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
from app.tools.doc_reader import DocReadError, IMAGE_EXTS, SUPPORTED_EXTS, extract_text_from_bytes
from app.tools.vision import describe_image

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_BYTES = 15 * 1024 * 1024      # 单文件 15MB
PREVIEW_CHARS = 200               # 返回给前端展示的开头预览
_DOC_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# 会话 -> 最近一次附件：追问时不用前端再传一遍（刷新、换标签页都不丢）
SESSION_DOC_PATH = os.path.join(BASE_DIR, "data", "session_docs.json")
SESSION_DOC_TTL = 12 * 3600
MAX_DOCS_PER_SESSION = 3


def _read_session_docs() -> dict:
    try:
        with open(SESSION_DOC_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_session_docs(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(SESSION_DOC_PATH), exist_ok=True)
        with open(SESSION_DOC_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[附件] 会话绑定写入失败：{e}")


def bind_session_doc(session_id: str, doc_id: str) -> None:
    """把这个文档挂到会话上；多次上传会累积，不会互相覆盖。"""
    if not session_id or not doc_id:
        return
    data = _read_session_docs()
    entry = data.get(session_id) or {}
    # 兼容旧格式（单文档）
    ids = list(entry.get("doc_ids") or ([entry["doc_id"]] if entry.get("doc_id") else []))
    if doc_id in ids:
        ids.remove(doc_id)
    ids.append(doc_id)
    # 一轮最多带 3 份，超出的丢掉最旧的
    ids = ids[-MAX_DOCS_PER_SESSION:]
    now = __import__("time").time()
    data[session_id] = {"doc_ids": ids, "doc_id": ids[-1], "ts": now}
    data = {k: v for k, v in data.items()
            if isinstance(v, dict) and now - v.get("ts", 0) < SESSION_DOC_TTL * 2}
    _write_session_docs(data)


def resolve_session_docs(session_id: str) -> list:
    """取这个会话挂着的所有附件（按上传顺序）；超过 TTL 视为失效。"""
    if not session_id:
        return []
    entry = _read_session_docs().get(session_id) or {}
    if __import__("time").time() - entry.get("ts", 0) > SESSION_DOC_TTL:
        return []
    ids = entry.get("doc_ids")
    if not ids and entry.get("doc_id"):
        ids = [entry["doc_id"]]
    return [x for x in (ids or []) if x]


def resolve_session_doc(session_id: str) -> str:
    """兼容旧调用：只要最近的一份。"""
    ids = resolve_session_docs(session_id)
    return ids[-1] if ids else ""


def unbind_session_doc(session_id: str) -> None:
    data = _read_session_docs()
    if data.pop(session_id, None) is not None:
        _write_session_docs(data)


class UploadRequest(BaseModel):
    filename: str
    content_base64: str
    session_id: str = ""


def _paths(doc_id: str):
    return os.path.join(UPLOAD_DIR, doc_id + ".json")
class DetachRequest(BaseModel):
    session_id: str


@router.post("/upload/detach", summary="解除会话与文档的绑定")
async def detach_document(request: Request, body: DetachRequest, auth: bool = Depends(verify_api_key)):
    unbind_session_doc(body.session_id)
    return {"code": 200, "message": "已解除绑定"}




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


@router.post("/upload", summary="上传文档或图片并解析为纯文本")
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
        if ext in IMAGE_EXTS:
            parsed = await __import__("asyncio").to_thread(describe_image, data, filename)
        else:
            parsed = await __import__("asyncio").to_thread(extract_text_from_bytes, data, filename)
    except DocReadError as e:
        return {"code": 422, "message": str(e)}
    except Exception as e:
        logger.error(f"文档解析异常: {e}")
        return {"code": 500, "message": f"文档解析失败：{e}"}

    doc_id = uuid.uuid4().hex

    # 记在会话上：下一步追问即使用户没带 attachment_id 也能找回这份文档
    if body.session_id:
        bind_session_doc(body.session_id, doc_id)
    record = {
        "doc_id": doc_id,
        "filename": filename,
        "kind": parsed["kind"],
        "chars": parsed["chars"],
        "warning": parsed["warning"],
        "via": parsed.get("model", ""),
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
