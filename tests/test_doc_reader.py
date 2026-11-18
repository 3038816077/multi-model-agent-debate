"""文档解析的回归测试。

不依赖外面的大文件：片场里现场拼一个最小 .pptx / .xlsx / .docx（本质都是 zip + XML），
验证解析器能按页/按表读出来，并且对不支持的格式给出人话报错。
"""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.doc_reader import (  # noqa: E402
    DocReadError,
    extract_text_from_bytes,
)

_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_X_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _slide_xml(*paragraphs: str) -> str:
    body = "".join(
        "<a:p><a:r><a:t>%s</a:t></a:r></a:p>" % p for p in paragraphs
    )
    return (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="%s"><p:cSld><p:spTree><p:sp><p:txBody>%s</p:txBody></p:sp>'
        "</p:spTree></p:cSld></p:sld>" % (_A_NS, body)
    )


def _make_pptx(path: str, slides) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for i, paragraphs in enumerate(slides, 1):
            z.writestr("ppt/slides/slide%d.xml" % i, _slide_xml(*paragraphs))


def test_pptx_reads_slides_in_order(tmp_path):
    path = os.path.join(str(tmp_path), "a.pptx")
    _make_pptx(path, [("第一页标题", "要点一"), ("第二页标题",)])
    with open(path, "rb") as f:
        result = extract_text_from_bytes(f.read(), "a.pptx")
    assert result["kind"] == "PPT"
    text = result["text"]
    assert "--- 第 1 页 ---" in text and "--- 第 2 页 ---" in text
    # 页码顺序不能乱：10 页以上时 slide2 不能排在 slide10 后面
    assert text.index("第一页标题") < text.index("第二页标题")


def test_pptx_slide_number_sorting(tmp_path):
    path = os.path.join(str(tmp_path), "b.pptx")
    _make_pptx(path, [("S%d" % i,) for i in range(1, 12)])  # 11 页，含两位数
    with open(path, "rb") as f:
        text = extract_text_from_bytes(f.read(), "b.pptx")["text"]
    assert text.index("S2") < text.index("S10")


def test_xlsx_reads_shared_strings(tmp_path):
    path = os.path.join(str(tmp_path), "a.xlsx")
    shared = (
        '<sst xmlns="%s"><si><t>月份</t></si><si><t>销量</t></si>'
        "<si><t>一月</t></si></sst>" % _X_NS
    )
    sheet = (
        '<worksheet xmlns="%s"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>120</v></c></row>'
        "</sheetData></worksheet>" % _X_NS
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
        z.writestr("xl/sharedStrings.xml", shared)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    with open(path, "rb") as f:
        result = extract_text_from_bytes(f.read(), "a.xlsx")
    assert result["kind"] == "Excel"
    assert "月份 | 销量" in result["text"]
    assert "一月 | 120" in result["text"]


def test_plain_text_autodetect():
    data = "中文内容".encode("utf-8")
    result = extract_text_from_bytes(data, "note.md")
    assert result["text"] == "中文内容"


def test_unsupported_extension_raises():
    try:
        extract_text_from_bytes(b"x", "a.xyz")
    except DocReadError as e:
        assert "暂不支持" in str(e)
    else:
        raise AssertionError("不支持的格式应该抛 DocReadError")


def test_broken_pptx_gives_human_message():
    try:
        extract_text_from_bytes(b"not a zip", "broken.pptx")
    except DocReadError as e:
        assert "损坏" in str(e) or "另存为" in str(e)
    else:
        raise AssertionError("损坏的 pptx 应该抛 DocReadError")


if __name__ == "__main__":
    import tempfile

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(d)
            except TypeError:
                fn()
        print("PASS", name)
    print("全部通过：%d 项" % len(fns))
