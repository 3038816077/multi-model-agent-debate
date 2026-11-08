import re
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

# ================= 👇 新增：前置参数校验 =================
def _validate_tool_args(tool_name: str, tool_args: dict) -> str | None:
    """根据工具定义的 JSON Schema 进行参数防呆校验"""
    # 从 TOOL_SCHEMAS 中找到对应的 schema
    schema = next((t["function"]["parameters"] for t in TOOL_SCHEMAS if t["function"]["name"] == tool_name), None)
    if not schema:
        return None
    
    required = schema.get("required", [])
    
    # 1. 必填项拦截
    for field in required:
        if field not in tool_args or tool_args[field] is None or str(tool_args[field]).strip() == "":
            return f"参数错误：缺少必填字段 [{field}] 或该字段为空"
    
    # 2. 空值拦截（针对非必填但传了纯空格字符串的）
    for field, value in tool_args.items():
        if isinstance(value, str) and value.strip() == "":
            return f"参数错误：字段 [{field}] 不能为空"
            
    return None

# ================= 👇 新增：后置结果清洗 =================
def _clean_tool_result(content: str) -> str:
    """清洗工具返回的脏数据，降低大模型 Token 消耗"""
    if not content:
        return ""
    content = content.strip() # 去除首尾空白
    content = re.sub(r'\n{3,}', '\n\n', content) # 压缩多余空行
    content = re.sub(r' {2,}', ' ', content)     # 压缩多余空格
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content) # 过滤不可见字符
    return content

# ================= 👇 升级：统一执行入口 =================
def execute_tool(tool_name: str, tool_args: dict, session_id: str = None) -> str:
    logger.info(f"[工具调用] 执行工具: {tool_name}, 参数: {tool_args}")
    
    if tool_name not in TOOLS_MAP:
        return f"错误：未知工具 {tool_name}"
    
    # 👇 第一步：前置参数校验（防呆设计，拦截后触发大模型自愈）
    validate_error = _validate_tool_args(tool_name, tool_args)
    if validate_error:
        logger.warning(f"[参数拦截] {validate_error}")
        return validate_error + "，请修正参数后重新调用"

    try:
        # 👇 第二步：真正执行工具
        raw_result = TOOLS_MAP[tool_name](**tool_args)
        logger.info(f"[工具返回] 原始结果: {raw_result}")
        
        # 👇 第三步：结果标准化清洗
        clean_result = _clean_tool_result(str(raw_result))
        
        # 👇 第四步：天气工具自动存入对应会话的缓存（存入清洗后的数据）
        if tool_name == "get_weather" and session_id:
            city = tool_args.get("city", "")
            if city and "天气：" in clean_result:
                if session_id not in WEATHER_CACHE:
                    WEATHER_CACHE[session_id] = {}
                WEATHER_CACHE[session_id][city] = clean_result
                
        return clean_result
    except Exception as e:
        logger.error(f"[工具异常] 工具执行异常：{str(e)}")
        return f"工具执行异常：{str(e)}"