"""文档解析：把用户上传的 Word / PDF / 文本文件转成纯文本。

· Word(.docx)  —— 用标准库解析（docx 本质是 zip + XML），不额外引入依赖
· PDF          —— 用 requirements 里已有的 pypdf
· txt/md/csv   —— 直接按文本读，自动尝试 utf-8 / gbk

对外只有两个函数：
    extract_text(path)            -> dict(text, chars, kind, warning)
    extract_text_from_bytes(...)  -> 同上（上传时内容在内存里，不用先落盘）
"""
import io
import os
import zipfile
from xml.etree import ElementTree as ET

# 单份文档最多保留的字符数，防止一个超大文件把内存和上下文撑爆
MAX_CHARS = 200_000

# 前端 accept 与后端白名单保持一致
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".log", ".json"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocReadError(Exception):
    """解析失败时抛出，message 是能直接展示给用户的中文说明。"""


# ==================== Word ====================
def _paragraph_text(p) -> str:
    """把一个 <w:p> 段落拼成字符串：w:t 取文字，w:tab 补制表符，w:br 补换行。"""
    buf = []
    for node in p.iter():
        if node.tag == _W + "t":
            buf.append(node.text or "")
        elif node.tag == _W + "tab":
            buf.append("\t")
        elif node.tag in (_W + "br", _W + "cr"):
            buf.append("\n")
    return "".join(buf)


def _docx_text(fp) -> str:
    try:
        with zipfile.ZipFile(fp) as z:
            if "word/document.xml" not in z.namelist():
                raise DocReadError("这不是有效的 .docx 文件（可能是老的 .doc 格式，请另存为 .docx 后再上传）")
            xml = z.read("word/document.xml")
    except zipfile.BadZipFile:
        raise DocReadError("文件已损坏或不是有效的 .docx（老的 .doc 格式请先另存为 .docx）")

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        raise DocReadError("Word 文档内部结构解析失败")

    body = root.find(_W + "body")
    source = body if body is not None else root

    lines = []
    for child in list(source):
        if child.tag == _W + "p":
            lines.append(_paragraph_text(child))
        elif child.tag == _W + "tbl":
            # 表格按 "单元格 | 单元格" 的纯文本还原
            for row in child.findall(_W + "tr"):
                cells = [_paragraph_text(c).strip().replace("\n", " ") for c in row.findall(_W + "tc")]
                lines.append(" | ".join(cells))
            lines.append("")
    return "\n".join(lines)


# ==================== PDF ====================
def _pdf_text(fp) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocReadError("服务端缺少 pypdf，无法解析 PDF，请执行：pip install pypdf")

    try:
        reader = PdfReader(fp)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                raise DocReadError("这个 PDF 有密码保护，请先解除加密再上传")
        pages = []
        for i, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                pages.append(f"--- 第 {i} 页 ---\n{text.strip()}")
        return "\n\n".join(pages)
    except DocReadError:
        raise
    except Exception as e:
        raise DocReadError(f"PDF 解析失败：{e}")


# ==================== 纯文本 ====================
def _plain_text(fp) -> str:
    data = fp.read() if hasattr(fp, "read") else fp
    if isinstance(data, bytes):
        for enc in ("utf-8", "gbk", "utf-16", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="ignore")
    return str(data)


# ==================== 对外入口 ====================
def _extract(fp, filename: str, size_hint: int = 0) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise DocReadError(f"暂不支持 {ext or '该'} 格式，目前支持：{'、'.join(sorted(SUPPORTED_EXTS))}")

    if ext == ".docx":
        text, kind = _docx_text(fp), "Word"
    elif ext == ".pdf":
        text, kind = _pdf_text(fp), "PDF"
    else:
        text, kind = _plain_text(fp), "文本"

    # 压掉三个以上的连续空行，读起来更紧凑，也省 token
    import re
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    warning = ""
    if not text:
        warning = "没有从文档里提取到文字（可能是扫描件/图片型 PDF，需要先做 OCR）"
        if ext == ".pdf":
            raise DocReadError(warning)
    elif len(text) > MAX_CHARS:
        warning = f"文档较长（{len(text)} 字），只保留了前 {MAX_CHARS} 字"
        text = text[:MAX_CHARS]

    return {"text": text, "chars": len(text), "kind": kind, "warning": warning}


def extract_text(path: str) -> dict:
    """从磁盘文件解析。"""
    filename = os.path.basename(path)
    with open(path, "rb") as fp:
        return _extract(fp, filename)


def extract_text_from_bytes(data: bytes, filename: str) -> dict:
    """从内存字节解析（上传场景，省一次落盘）。"""
    return _extract(io.BytesIO(data), filename)
