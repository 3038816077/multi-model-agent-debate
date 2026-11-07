from app.tools.registry import register_tool
from app.db.vector_store import hybrid_search
from app.core.config import client, MODEL_NAME, RAG_FINAL_TOP_K
from app.core.logger import logger

def _rewrite_query(query: str) -> str:
    """将用户口语化提问改写为标准检索关键词，失败自动返回原句"""
    try:
        # 采用极简 Prompt，约束大模型只输出关键词
        prompt = f"""请将用户问题提炼为适合知识库检索的核心关键词，用空格分隔。
只输出关键词，不要解释、不要多余内容、不要标点符号。
用户问题：{query}"""
        
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 温度设低，保证输出稳定
            max_tokens=50     # 限制长度，防止废话
        )
        rewritten = resp.choices[0].message.content.strip()
        
        if rewritten and len(rewritten) > 0:
            logger.info(f"[Query改写] 原query: '{query}' -> 改写为: '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        logger.warning(f"[Query改写] 失败，降级使用原query: {str(e)}")
        return query

@register_tool(
    name="search_knowledge",
    description="查询产品使用说明、功能介绍、操作规则、常见问题等固定文档内容。涉及内部规则和文档类问题时调用此工具。",
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "用户的问题原文"}}, "required": ["query"]}
)
def search_knowledge(query: str) -> str:
    if not query:
        return "知识库检索失败：查询内容为空"

  

    # 1. 查询改写：将口语化提问转为精准关键词
    search_query = _rewrite_query(query)
    
    # 2. 执行混合检索 + 精排（使用改写后的 search_query，并使用配置的 Top K）
    results = hybrid_search(search_query, top_k=RAG_FINAL_TOP_K)
    
    if not results:
        return "知识库未检索到相关内容。"
    return "知识库检索结果：\n" + "\n---\n".join(results)