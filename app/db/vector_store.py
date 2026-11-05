import os
import jieba
import re
from pypdf import PdfReader

# 👇 1. 动态设置离线模式（必须在导入 HuggingFace 相关库之前设置）
# 如果你在 .env 里配了 HF_HUB_OFFLINE=1，本地就会秒开；否则默认允许联网
hf_offline = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["HF_HUB_OFFLINE"] = hf_offline
os.environ["TRANSFORMERS_OFFLINE"] = hf_offline

# 👇 2. 清理 jieba 的 DEBUG 日志
jieba.setLogLevel(jieba.logging.INFO)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.core.logger import logger

ENSEMBLE_RETRIEVER = None
KNOWLEDGE_DIR = ""

def load_file_content(filepath: str) -> str:
    """统一读取 txt/md/pdf 格式文档"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # 清理 PDF 提取可能出现的多余空行
        return re.sub(r'\n{3,}', '\n\n', text)
    return ""

def _chinese_tokenizer(text: str) -> list:
    """BM25 中文分词器，解决原生空格分词不适配中文的问题"""
    return list(jieba.cut(text))

def init_knowledge_base(base_dir: str):
    """初始化混合检索知识库：向量检索 + BM25关键词检索 + RRF融合"""
    global ENSEMBLE_RETRIEVER, KNOWLEDGE_DIR
    KNOWLEDGE_DIR = os.path.join(base_dir, "knowledge")

    if not os.path.exists(KNOWLEDGE_DIR):
        os.mkdir(KNOWLEDGE_DIR)
        logger.info("已创建知识库目录，请放入 .txt/.md/.pdf 文档")
        return

    docs = []
    for filename in os.listdir(KNOWLEDGE_DIR):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        if filename.endswith((".txt", ".md", ".pdf")):
            try:
                content = load_file_content(filepath)
                if content.strip():
                    docs.append(content)
            except Exception as e:
                logger.error(f"读取文档 {filename} 失败: {str(e)}")

    if not docs:
        logger.info("知识库目录为空，跳过向量库初始化")
        return

    # 语义切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    split_docs = splitter.create_documents(docs)

    # 3. 向量检索器 (Dense)
    is_offline = (hf_offline == "1")
    logger.info(f"正在加载向量模型 (离线模式: {is_offline})...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu", 
            "local_files_only": is_offline  # 👈 动态应用离线参数
        }
    )
    vector_db = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 4. BM25 关键词检索器 (Sparse)
    bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=_chinese_tokenizer)
    bm25_retriever.k = 10

    # 5. RRF 融合检索器
    # 权重：BM25 占 30%，FAISS 向量占 70%（可根据实际业务调整）
    ENSEMBLE_RETRIEVER = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )

    logger.info(f"知识库初始化完成，共 {len(split_docs)} 个文本块，混合检索模式已启用")

def hybrid_search(query: str, top_k: int = 3):
    """对外暴露的混合检索接口"""
    if not ENSEMBLE_RETRIEVER:
        return []
    try:
        # 触发混合检索
        results = ENSEMBLE_RETRIEVER.invoke(query)
        
        # 去重逻辑（避免向量和关键词召回了同一个文本块）
        seen = set()
        unique_docs = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc.page_content)
                if len(unique_docs) >= top_k:
                    break
        return unique_docs
    except Exception as e:
        logger.error(f"混合检索失败: {str(e)}")
        return []