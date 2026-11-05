from app.core.logger import logger

TOOL_SCHEMAS = []
TOOLS_MAP = {}
WEATHER_CACHE = {}

def register_tool(name: str, description: str, parameters: dict):
    def decorator(func):
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}
        })
        TOOLS_MAP[name] = func
        return func
    return decorator

def execute_tool(tool_name: str, tool_args: dict, session_id: str = None) -> str:
    logger.info(f"[工具调用] 执行工具: {tool_name}, 参数: {tool_args}")
    if tool_name not in TOOLS_MAP:
        return f"错误：未知工具 {tool_name}"
    
    try:
        result = TOOLS_MAP[tool_name](**tool_args)
        logger.info(f"[工具返回] 结果: {result}")
        
                # 天气工具自动存入对应会话的缓存
        if tool_name == "get_weather" and session_id:
            city = tool_args.get("city", "")
            # 👇 加一个判断：只有真正查到了天气，才存入缓存
            if city and "天气：" in result:
                if session_id not in WEATHER_CACHE:
                    WEATHER_CACHE[session_id] = {}
                WEATHER_CACHE[session_id][city] = result
        return result
    except Exception as e:
        logger.info(f"[工具异常] 工具执行异常：{str(e)}")
        return f"工具执行异常：{str(e)}"