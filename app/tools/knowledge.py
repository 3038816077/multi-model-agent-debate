from app.tools.registry import register_tool
from app.db.vector_store import hybrid_search

@register_tool(
    name="search_knowledge",
    description="查询产品使用说明、功能介绍、操作规则、常见问题等固定文档内容。涉及内部规则和文档类问题时调用此工具。",
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "用户的问题原文"}}, "required": ["query"]}
)
def search_knowledge(query: str) -> str:
    # 调用 hybrid_search，注意参数名是 top_k
    results = hybrid_search(query, top_k=3)
    if not results:
        return "知识库未检索到相关内容。"
    return "知识库检索结果：\n" + "\n---\n".join(results)