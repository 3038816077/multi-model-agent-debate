FROM python:3.10-slim

WORKDIR /app

# 注意：这里的 URL 必须是没有方括号的纯文本
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_OFFLINE=0

# 安装基础编译依赖（faiss-cpu 需要）
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用Docker缓存，改代码不用重新装包
COPY requirements.txt .
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

EXPOSE 8000

CMD ["python", "run.py"]