import requests
from app.core.config import QWEATHER_API_KEY
from app.core.logger import logger
from app.tools.registry import register_tool

@register_tool(
    name="get_weather",
    description="查询指定城市的实时天气情况。当用户询问任何城市的天气时，必须先调用此工具获取真实数据，严禁直接回答或编造。",
    parameters={"type": "object", "properties": {"city": {"type": "string", "description": "要查询的城市名称，例如：北京"}}, "required": ["city"]}
)
def get_weather(city: str) -> str:
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {"key": QWEATHER_API_KEY, "location": city, "language": "zh-Hans", "unit": "c"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        logger.info(f"[天气接口原始返回] {data}")
        if "status_code" in data and data["status_code"] != 0:
            return f"查询失败：{data.get('status', '未知错误')}"
        if not data.get("results"):
            return f"暂未查询到「{city}」的天气信息，请确认城市名称"
        
        result = data["results"][0]
        now = result["now"]
        return f"{result['location']['name']}天气：{now.get('text', '未知')}，气温 {now.get('temperature', '未知')}℃，{now.get('wind_direction', '无持续风向')} {now.get('wind_scale', '0')}级"
    except requests.exceptions.Timeout:
        return "天气接口请求超时，请稍后重试"
    except Exception as e:
        return f"天气查询异常：{str(e)}"