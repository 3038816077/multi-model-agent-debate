import re
from app.core.config import MODEL_NAME
# 👇 引入动态 LLM 工厂
from app.core.llm_factory import get_llm_client

def clean_latex_text(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = text.replace("`", "").replace("$", "")
    text = text.replace("\\times", "×").replace("\\div", "÷").replace("\\cdot", "·").replace("\\pm", "±")
    text = re.sub(r'\\', '', text)
    return text.strip()

def clean_weather_reply(text: str) -> str:
    text = re.sub(r'[，,。.]\s*如需.*$', '', text, flags=re.S)
    text = re.sub(r'[，,。.]\s*是否需要.*$', '', text, flags=re.S)
    text = re.sub(r'[，,。.]\s*我可以.*$', '', text, flags=re.S)
    text = re.sub(r'[，,。.]\s*需要.*请告知.*$', '', text, flags=re.S)
    text = re.sub(r'^(您提供的.*?[：:]|根据.*?信息[，,]|已知.*?显示[，,]?)', '', text)
    return text.strip().rstrip('，,。.').strip()

def generate_dialog_summary(old_messages: list) -> str:
    prompt = "请将以下对话历史压缩成一段精简摘要，只保留关键的工具查询结果、用户核心需求和重要结论，去掉冗余对话和寒暄内容，控制在120字以内。\n\n对话历史：\n"
    dialog_text = ""
    for msg in old_messages:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role in ["user", "assistant", "tool"]:
            dialog_text += f"{role}: {content}\n"
    
    # 👇 替换为动态获取 Client
    resp = get_llm_client().chat.completions.create(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": prompt + dialog_text}]
    )
    return "【历史对话摘要】" + resp.choices[0].message.content

def extract_long_term_memory(messages: list, existing_memory: str = "") -> str:
    prompt = """请从以下对话中提取需要长期记住的用户关键信息，包括：常查询的城市、固定需求、用户偏好、重要结论。
要求：
1. 只提取确定的事实信息，不要推测
2. 用简短的要点式输出，每条一行
3. 合并已有的旧记忆，去重保留最新信息
4. 如果没有值得长期记住的信息，就返回空字符串
旧记忆：{old_memory}
最新对话：
{dialog}"""
    recent_msgs = messages[-6:] if len(messages) >= 6 else messages
    dialog_text = ""
    for msg in recent_msgs:
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role in ["user", "assistant"]:
            dialog_text += f"{role}: {content}\n"
    
    if not dialog_text:
        return existing_memory
    
    # 👇 替换为动态获取 Client
    resp = get_llm_client().chat.completions.create(
        model=MODEL_NAME, 
        messages=[{"role": "user", "content": prompt.format(old_memory=existing_memory, dialog=dialog_text)}]
    )
    new_memory = resp.choices[0].message.content.strip()
    return new_memory if new_memory != "空" else existing_memory