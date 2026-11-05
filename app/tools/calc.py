import re
from datetime import datetime
from app.tools.registry import register_tool

@register_tool(
    name="calculate_expression",
    description="通用数学计算工具，支持加减乘除、括号混合运算，适用于所有数值计算场景",
    parameters={"type": "object", "properties": {"expression": {"type": "string", "description": "待计算的数学表达式，例如：(12+78)*3"}}, "required": ["expression"]}
)
def calculate_expression(expression: str) -> str:
    if not re.fullmatch(r'[\d+\-*/().\s]+', expression):
        return "错误：表达式包含非法字符，仅支持数字与 + - * / ( ) 运算符"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果：{expression} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except SyntaxError:
        return "错误：表达式语法不正确，请检查括号和运算符格式"
    except Exception as e:
        return f"计算失败：{str(e)}"

@register_tool(
    name="get_current_time",
    description="获取实时准确的当前日期、北京时间和星期几。模型没有实时时间能力，所有涉及时间、日期、星期的提问，必须调用此工具，禁止自行回答。",
    parameters={"type": "object", "properties": {}, "required": []}
)
def get_current_time() -> str:
    now = datetime.now()
    week_list = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')},{week_list[now.weekday()]}"