import os
import jieba
import re
from pypdf import PdfReader
from app.core.config import (
    RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_RECALL_TOP_K, RAG_FINAL_TOP_K,
    USE_SEMANTIC_CHUNKING, SEMANTIC_CHUNK_THRESHOLD # 👈 引入新增配置
)
# 👇 引入 Document 对象
from langchain_core.documents import Document

# 👇 1. 动态读取离线配置
hf_offline = os.getenv("HF_HUB_OFFLINE", "0")
os.environ["HF_HUB_OFFLINE"] = hf_offline
os.environ["TRANSFORMERS_OFFLINE"] = hf_offline

# 👇 2. 网络环境处理
if hf_offline == "1":
    os.environ["HF_ENDPOINT"] = ""
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["ALL_PROXY"] = ""
    os.environ["all_proxy"] = ""
else:
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

jieba.setLogLevel(jieba.logging.INFO)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from app.core.logger import logger

ENSEMBLE_RETRIEVER = None
RERANKER = None
KNOWLEDGE_DIR = ""

def load_file_content(filepath: str) -> str:
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
        return re.sub(r'\n{3,}', '\n\n', text)
    return ""

def _chinese_tokenizer(text: str) -> list:
    return list(jieba.cut(text))

def init_knowledge_base(base_dir: str):
    global ENSEMBLE_RETRIEVER, RERANKER, KNOWLEDGE_DIR
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
                    doc = Document(page_content=content, metadata={"source": filename})
                    docs.append(doc)
            except Exception as e:
                logger.error(f"读取文档 {filename} 失败: {str(e)}")

    if not docs:
        logger.info("知识库目录为空，跳过向量库初始化")
        return

    is_offline = (hf_offline == "1")
    
    # 👇 提前加载向量模型（因为语义分块需要它，固定分块也需要它建库）
    logger.info(f"正在加载向量模型 (离线模式: {is_offline})...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "local_files_only": is_offline}
    )

    # 👇 核心：根据配置选择分块策略
    if USE_SEMANTIC_CHUNKING:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            logger.info(f"【RAG】正在使用 语义分块 (Semantic) 策略，阈值: {SEMANTIC_CHUNK_THRESHOLD}...")
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SEMANTIC_CHUNK_THRESHOLD * 100
            )
            split_docs = splitter.split_documents(docs)
        except Exception as e:
            logger.error(f"语义分块失败，自动降级为固定分块: {e}")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP, separators=["\n\n", "\n", "。", "；", " ", ""]
            )
            split_docs = splitter.split_documents(docs)
    else:
        logger.info("【RAG】正在使用 字符硬切 (Recursive) 策略...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP, separators=["\n\n", "\n", "。", "；", " ", ""]
        )
        split_docs = splitter.split_documents(docs)

    # 1. 向量检索器
    vector_db = FAISS.from_documents(split_docs, embeddings)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": RAG_RECALL_TOP_K})

    # 2. BM25 关键词检索器
    bm25_retriever = BM25Retriever.from_documents(split_docs, preprocess_func=_chinese_tokenizer)
    bm25_retriever.k = RAG_RECALL_TOP_K

    # 3. RRF 融合检索器
    ENSEMBLE_RETRIEVER = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )

    # 4. Rerank 重排序模型
    logger.info(f"正在加载 Rerank 重排序模型...")
    try:
        RERANKER = CrossEncoder('BAAI/bge-reranker-base', max_length=512, local_files_only=is_offline)
        logger.info("Rerank 模型加载成功！")
    except Exception as e:
        logger.error(f"Rerank 模型加载失败，将退化为纯混合检索: {str(e)}")

    logger.info(f"知识库初始化完成，共 {len(split_docs)} 个文本块，【混合检索+精排+溯源】已启用")

def hybrid_search(query: str, top_k: int = RAG_FINAL_TOP_K):
    # ... (这里的 hybrid_search 保持你刚才的代码完全不变) ...
    if not ENSEMBLE_RETRIEVER:
        return []
    try:
        results = ENSEMBLE_RETRIEVER.invoke(query)
        seen = set()
        unique_docs = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
                
        if RERANKER and len(unique_docs) > 1:
            pairs = [[query, doc.page_content] for doc in unique_docs]
            scores = RERANKER.predict(pairs)
            scored_docs = list(zip(unique_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            unique_docs = [doc for doc, score in scored_docs]
            
        final_results = []
        for doc in unique_docs[:top_k]:
            source = doc.metadata.get("source", "未知文档")
            final_results.append(f"【来源文件: {source}】\n{doc.page_content}")
            
        return final_results
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        return []