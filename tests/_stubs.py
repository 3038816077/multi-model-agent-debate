"""单元测试用的轻量替身。

为什么要这个
------------
我们的单元测试只验证纯逻辑（引文校验、立场分组、文档解析、鉴权、备份），
但它们 import 的模块会顺带拉起模型运行时：torch / faiss / langchain / jieba。

让 CI 为了跑 60 个纯逻辑用例去装 2GB 的 torch 不值得，也不稳定。
所以这里把这些「运行时依赖」替换成空实现——只满足导入，不参与断言。

真实依赖的验证放在另一个地方：Docker 构建的最后一步会执行
`python -c "import app.main"`，装不全就会在构建阶段炸掉。
"""
import sys
import types


class _Any:
    """什么都能接受、什么都能返回的占位对象。"""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


class _StateGraph:
    """langgraph.graph.StateGraph 的最小替身，支持链式建图。"""

    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *a, **k):
        return self

    def set_entry_point(self, *a, **k):
        return self

    def add_conditional_edges(self, *a, **k):
        return self

    def add_edge(self, *a, **k):
        return self

    def compile(self, *a, **k):
        return _Any()


def _install(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install():
    """把重依赖替换成替身；重复调用无副作用。"""
    if sys.modules.get("_dsh_stubs"):
        return
    sys.modules["_dsh_stubs"] = types.ModuleType("_dsh_stubs")

    # jieba：向量库模块在导入时就要 setLogLevel
    jieba = _install("jieba", setLogLevel=lambda *a, **k: None,
                     lcut=lambda text, *a, **k: list(text or ""),
                     cut=lambda text, *a, **k: list(text or ""))
    jieba.logging = types.SimpleNamespace(INFO=20, DEBUG=10, WARNING=30, ERROR=40)

    # langchain 家族
    class _Doc:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    _install("langchain_core")
    _install("langchain_core.documents", Document=_Doc)
    _install("langchain_text_splitters", RecursiveCharacterTextSplitter=_Any)
    _install("langchain_community")
    _install("langchain_community.vectorstores", FAISS=_Any)
    _install("langchain_community.retrievers", BM25Retriever=_Any)
    _install("langchain_huggingface", HuggingFaceEmbeddings=_Any)
    _install("langchain_classic")
    _install("langchain_classic.retrievers", EnsembleRetriever=_Any)

    # 向量与重排模型
    _install("sentence_transformers", SentenceTransformer=_Any, CrossEncoder=_Any)

    # 图编排
    _install("langgraph")
    _install("langgraph.graph", StateGraph=_StateGraph, END="__end__")
