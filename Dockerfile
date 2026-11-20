FROM python:3.10-slim

WORKDIR /app

# 国内默认用清华源；CI（海外 runner）用 --build-arg 覆盖成 pypi.org
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_OFFLINE=0

# 👇 关键修改：替换系统源为国内源，解决 apt-get 极慢的问题
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用Docker缓存，改代码不用重新装包
COPY requirements.txt .
RUN pip install --upgrade pip -i "$PIP_INDEX_URL" --retries 5 --timeout 120 \
    && pip install -r requirements.txt -i "$PIP_INDEX_URL" --retries 5 --timeout 120 \
    && pip cache purge || true

# 复制项目代码
COPY . .

# 构建期就验证依赖装得全：import 失败会立刻让镜像构建失败，
# 而不是等容器跑起来才在日志里看到 ImportError
RUN python -c "import app.main; print('app import OK')"

EXPOSE 8000

CMD ["python", "run.py"]