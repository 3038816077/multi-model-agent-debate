"""多文件对比的回归测试。

守三件事：
1. 多份文档必须带编号标签，模型才分得清哪句来自哪份
2. 会话绑定要累积（上传第二份不能把第一份顶掉），但最多 3 份
3. 请求里同时给 attachment_id 和 attachment_ids 时不能重复注入
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.api.chat as chat  # noqa: E402
import app.api.upload as up_mod  # noqa: E402


class _Req:
    def __init__(self, attachment_id=None, attachment_ids=None):
        self.attachment_id = attachment_id
        self.attachment_ids = attachment_ids


def test_request_doc_ids_merges_and_dedupes():
    req = _Req(attachment_id="a", attachment_ids=["a", "b", None, "b"])
    assert chat.request_doc_ids(req) == ["a", "b"]


def test_request_doc_ids_caps_at_three():
    req = _Req(attachment_ids=["a", "b", "c", "d"])
    assert len(chat.request_doc_ids(req)) == chat.MAX_DOCS_PER_TURN == 3


def test_request_doc_ids_empty():
    assert chat.request_doc_ids(_Req()) == []


def _with_fake_docs(docs, fn):
    """临时替换 load_document，避免测试依赖真实上传目录。"""
    original = chat.load_document
    chat.load_document = lambda doc_id: docs.get(doc_id)
    try:
        return fn()
    finally:
        chat.load_document = original


def test_single_document_has_no_numbering():
    docs = {"a": {"filename": "方案.docx", "kind": "Word", "chars": 4, "text": "正文内容"}}
    msg = _with_fake_docs(docs, lambda: chat._documents_message(["a"]))
    assert msg and msg["role"] == "system"
    assert "【文档：方案.docx" in msg["content"]
    assert "文档1" not in msg["content"]


def test_multiple_documents_are_numbered_with_comparison_hint():
    docs = {
        "a": {"filename": "旧版合同.docx", "kind": "Word", "chars": 4, "text": "甲方是张三"},
        "b": {"filename": "新版合同.docx", "kind": "Word", "chars": 4, "text": "甲方是李四"},
    }
    msg = _with_fake_docs(docs, lambda: chat._documents_message(["a", "b"]))
    assert msg is not None
    content = msg["content"]
    assert "【文档1：旧版合同.docx" in content
    assert "【文档2：新版合同.docx" in content
    assert "甲方是张三" in content and "甲方是李四" in content
    # 多份时必须有对比指引
    assert "对比" in content
    # 顺序不能乱
    assert content.index("文档1") < content.index("文档2")


def test_missing_document_is_skipped():
    docs = {"a": {"filename": "A.docx", "kind": "Word", "chars": 2, "text": "甲"}}
    msg = _with_fake_docs(docs, lambda: chat._documents_message(["a", "不存在的id"]))
    assert msg is not None and "文档2" not in msg["content"]


def test_all_documents_missing_returns_none():
    msg = _with_fake_docs({}, lambda: chat._documents_message(["x", "y"]))
    assert msg is None


def test_session_binding_accumulates_and_caps():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        original_path = up_mod.SESSION_DOC_PATH
        up_mod.SESSION_DOC_PATH = os.path.join(d, "session_docs.json")
        try:
            up_mod.bind_session_doc("s1", "d1")
            up_mod.bind_session_doc("s1", "d2")
            assert up_mod.resolve_session_docs("s1") == ["d1", "d2"]
            assert up_mod.resolve_session_doc("s1") == "d2"
            up_mod.bind_session_doc("s1", "d3")
            up_mod.bind_session_doc("s1", "d4")
            ids = up_mod.resolve_session_docs("s1")
            assert len(ids) == up_mod.MAX_DOCS_PER_SESSION == 3
            assert ids == ["d2", "d3", "d4"]
            # 重复上传同一份不该产生重复
            up_mod.bind_session_doc("s1", "d4")
            assert up_mod.resolve_session_docs("s1") == ["d2", "d3", "d4"]
            # 解绑
            up_mod.unbind_session_doc("s1")
            assert up_mod.resolve_session_docs("s1") == []
        finally:
            up_mod.SESSION_DOC_PATH = original_path


def test_legacy_single_doc_binding_is_readable():
    import json, tempfile

    with tempfile.TemporaryDirectory() as d:
        original_path = up_mod.SESSION_DOC_PATH
        path = os.path.join(d, "session_docs.json")
        up_mod.SESSION_DOC_PATH = path
        try:
            import time
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"s9": {"doc_id": "old123", "ts": time.time()}}, f)
            assert up_mod.resolve_session_docs("s9") == ["old123"]
        finally:
            up_mod.SESSION_DOC_PATH = original_path


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("全部通过：%d 项" % len(fns))
