# 🤖 多工具智能助手 Agent (Hybrid RAG + LangGraph)

基于 **LangGraph + FastAPI + SQLite + FAISS** 从零搭建的企业级多工具智能体。项目包含完整的 ReAct 闭环、混合检索（Hybrid RAG）架构、并发状态隔离、多级记忆与会话持久化，支持 Docker 一键容器化部署。

## ✨ 核心工程亮点

- 🧠 **意图前置路由与防死循环**：设计正则匹配的 Intent Router，对时间/计算等确定性场景进行后端兜底隔离；在 LLM 节点动态判断 `tool_choice`（首次 `required` 强控，二次 `auto` 恢复），彻底解决大模型“幻觉不调工具”与“工具无限死循环”的痛点。
- 🔍 **混合检索架构 (Hybrid RAG)**：针对纯向量检索对专有名词、设备型号召回率极低的痛点，引入 **BM25（稀疏检索） + FAISS（稠密检索）**；基于 LangChain `EnsembleRetriever` 实现 **RRF（倒数排名融合）算法**，并接入 `jieba` 中文分词，支持 PDF/Markdown 多格式解析。
- 💾 **持久化与会话管理 (State Management)**：摒弃传统内存字典，引入 SQLAlchemy + SQLite 实现对话状态持久化。设计 Session 与 Message 双层表，结合后台长期记忆提取（Memory Extraction），实现服务重启及多进程环境下的跨会话状态无缝恢复。
- ⚡ **并发安全与防踩踏优化**：废弃全局状态字典，基于 LangGraph State 的 `input_len` 增量截取 `final_messages`，实现并发请求下执行链路（思考过程）的无状态提取，彻底解决多用户并发导致的上下文污染与数据错乱。
- 🏗️ **DDD 分层与工程化交付**：采用标准领域分层架构（`api` / `agent` / `core` / `db` / `tools`）。提供统一全局异常拦截，支持 Docker 多阶段构建与数据卷挂载一键部署。

## 🛠️ 技术栈
- **核心框架**：FastAPI + Uvicorn + SQLAlchemy
- **Agent 编排**：LangGraph (StateGraph) + ReAct 范式
- **大语言模型**：通义千问 Qwen-Plus (OpenAI 兼容协议)
- **检索增强 (RAG)**：FAISS + BM25 + Jieba + HuggingFace Embedding
- **数据库**：SQLite (持久化) + 内存缓存 (天气/高频工具)
- **部署运维**：Docker + Docker Compose

## 📁 项目结构
```text
agent-demo/
├── app/
│   ├── api/          # 路由层：鉴权、意图拦截、思考过程提取
│   ├── agent/        # 核心层：LangGraph 状态机、LLM 节点、Tool 节点
│   ├── core/         # 基础层：全局配置、日志、文本清洗
│   ├── db/           # 数据层：SQLite 模型、CRUD、向量库 (FAISS+BM25)
│   └── tools/        # 工具层：天气、计算、时间、RAG 检索
├── knowledge/        # 本地知识库目录 (支持 .txt/.md/.pdf)
├── static/           # 前端交互页面
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py            # 启动入口
```
###  🚀 快速启动
## 方式一：本地运行
安装依赖：pip install -r requirements.txt

配置环境变量：将 env.txt 重命名为 .env，并填入你的 API Key。
(注：首次启动会联网下载约 80MB 的向量化模型，下载完成后可在 .env 中设置 HF_HUB_OFFLINE=1 开启离线秒开模式)

启动服务：python run.py

访问页面：http://127.0.0.1:8000

## 方式二：Docker 一键部署
bash
# 确保本地已配置好 .env 文件
docker compose up -d --build
🎯 效果演示
多步链式调用：输入 北京、上海、厦门天气，Agent 会自动并行调用 3 次天气工具并汇总回答。

专有名词检索：在 knowledge 放入包含特定暗号的文档，提问时依靠 BM25 混合检索精准召回。

思考过程可视化：前端可展开查看完整的“思考 - 调用 - 观察 - 回答”链路。
