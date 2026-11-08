import os
import httpx
from app.tools.registry import register_tool
from app.core.logger import logger

@register_tool(
    name="search_web",
    description="当用户询问实时新闻、近期事件、未知的最新资讯，或要求联网搜索时，必须调用此工具获取最新信息。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["query"]
    }
)
def search_web(query: str) -> str:
    try:
        api_key = os.getenv("BOCHA_API_KEY")
        if not api_key:
            return "联网搜索失败：未配置 BOCHA_API_KEY"

        logger.info(f"[博查搜索] Query: {query}")
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": 3
        }
        with httpx.Client(timeout=30) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code != 200:
                return f"博查搜索失败，状态码：{response.status_code}"

            result = response.json()
            # 根据博查返回结构提取网页结果
            webpages = result.get("data", {}).get("webPages", {}).get("value", [])
            if not webpages:
                return "未搜索到相关结果。"

            formatted = []
            for i, page in enumerate(webpages, 1):
                formatted.append(
                    f"[{i}] {page.get('name', '无标题')}\n"
                    f"摘要：{page.get('snippet', '无摘要')}\n"
                    f"链接：{page.get('url', '无链接')}"
                )
            return "联网搜索结果：\n" + "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"[博查搜索] 失败: {str(e)}")
        return f"联网搜索失败：{str(e)}"