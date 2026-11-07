
# 🤖 多工具智能助手 Agent (Hybrid RAG + LangGraph)

基于 LangGraph + FastAPI + SQLite + FAISS 从零搭建的多工具智能体。项目完整实现了 ReAct 闭环、两阶段混合检索（粗排+精排）、多工具线程池并发、并发状态隔离、多级记忆与会话持久化，并深度集成了接口鉴权、限流防护与工具错误自愈机制。支持 Docker 多阶段构建与一键容器化交付
## ✨ 核心工程亮点
- ⚡ 多工具线程池并发调度：针对工具调用属于 IO 密集型任务的特点，在 tool_node 中引入 ThreadPoolExecutor。当大模型一次性发起多个 Function Calling 时，系统并发请求第三方接口，多任务总耗时缩短 60% 以上，并通过 Future 机制保证上下文顺序绝对保真。

- 🔍 两阶段混合检索架构 (Two-Stage Hybrid RAG)：针对纯向量检索易召回冗余废话的痛点，先通过 BM25+FAISS 双路召回+RRF 融合进行粗排（Top 10），再引入轻量级 Cross-Encoder (BAAI/bge-reranker-base) 进行深度交互精排（Top 3）。有效过滤噪声，极大提升上下文纯度。
- 🧠 **意图前置路由与防死循环**：设计正则匹配的 Intent Router，对时间/计算等确定性场景进行后端兜底隔离；在 LLM 节点动态判断 `tool_choice`（首次 `required` 强控，二次 `auto` 恢复），彻底解决大模型“幻觉不调工具”与“工具无限死循环”的痛点。
- 🔍 **混合检索架构 (Hybrid RAG)**：针对纯向量检索对专有名词、设备型号召回率极低的痛点，引入 **BM25（稀疏检索） + FAISS（稠密检索）**；基于 LangChain `EnsembleRetriever` 实现 **RRF（倒数排名融合）算法**，并接入 `jieba` 中文分词，支持 PDF/Markdown 多格式解析。
- 🛡️ 安全防护与容错机制：新增 API Key 接口鉴权与基于 slowapi 的单 IP 限流（30次/分钟），防止恶意刷量；在 tool_node 中设计了工具异常自愈机制，当接口超时或参数错误时，将 Error 日志回流给大模型，引导其自主修正参数重试（最多2次），大幅提升复杂任务成功率。
- 💾 **持久化与会话管理 (State Management)**：摒弃传统内存字典，引入 SQLAlchemy + SQLite 实现对话状态持久化。设计 Session 与 Message 双层表，结合后台长期记忆提取（Memory Extraction），实现服务重启及多进程环境下的跨会话状态无缝恢复。
- ⚡ **并发安全与防踩踏优化**：废弃全局状态字典，基于 LangGraph State 的 `input_len` 增量截取 `final_messages`，实现并发请求下执行链路（思考过程）的无状态提取，彻底解决多用户并发导致的上下文污染与数据错乱。
- 🏗️ **DDD 分层与工程化交付**：采用标准领域分层架构（`api` / `agent` / `core` / `db` / `tools`）。提供统一全局异常拦截，支持 Docker 多阶段构建与数据卷挂载一键部署。

## 🛠️ 技术栈

- **核心框架**：FastAPI + Uvicorn + SQLAlchemy
- **Agent 编排**：LangGraph (StateGraph) + ReAct 范式 + 线程池并发
- **大语言模型**：通义千问 Qwen-Plus (OpenAI 兼容协议)
- **检索增强 (RAG)**：FAISS + BM25 + Jieba + HuggingFace Embedding + BGE-Reranker 精排
- **安全防护**：SlowAPI (限流) + 自定义 API Key 鉴权
- **数据库**：SQLite (持久化) + 内存缓存 (天气/高频工具)
- **部署运维**：Docker + Docker Compose

## 📁 项目结构

```text
agent-demo/
├── app/
│   ├── api/          # 路由层：鉴权、意图拦截、思考过程提取
│   ├── agent/ # 核心层：LangGraph 状态机、LLM 节点、并发 Tool 节点
│   ├── core/         # 基础层：全局配置、日志、文本清洗
│   ├── db/ # 数据层：SQLite 模型、CRUD、向量库 (FAISS+BM25+精排) 
│   ├── middleware/ # 中间件层：API Key 鉴权与 IP 限流器
│   └── tools/        # 工具层：天气、计算、时间、RAG 检索
├── knowledge/        # 本地知识库目录 (支持 .txt/.md/.pdf)
├── static/           # 前端交互页面
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py            # 启动入口
```
## ⚠️ 网络异常无法下载模型怎么办？（离线兜底方案）
由于国内网络访问 HuggingFace 官方受限，如果你在首次启动时卡在 `正在加载向量模型`，请直接使用我们提供的离线模型包：
1. 点击去 **[Releases](https://github.com/3038816077/multi-tool-agent-demo/releases/tag/v1.0.0)** 页面下载 `all-MiniLM-L6-v2.zip`。
2. 将压缩包解压，把里面的 `hub` 文件夹放到你的电脑缓存目录：
   - Windows: `C:\Users\你的用户名\.cache\huggingface\`
   - Mac/Linux: `~/.cache/huggingface/`
3. 打开 `.env` 文件，设置 `HF_HUB_OFFLINE=1`，重新启动服务即可实现秒级冷启动。



## 🚀 快速启动

### 方式一：本地运行

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **配置环境变量**：将 `env.txt` 重命名为 `.env`，填入你的大模型与天气 API Key
   > *(注：如需测试接口鉴权，请在 .env 中配置 API_KEY="my-secret-key-123"；首次启动请保持 HF_HUB_OFFLINE=0 联网下载模型)*
3. **启动服务**：
   ```bash
   python run.py
   ```
4. **访问页面**：[http://127.0.0.1:8000](http://127.0.0.1:8000)

### 方式二：Docker 一键部署

1. 确保本地已配置好 `.env` 文件。
2. 在项目根目录执行构建并后台启动：
   ```bash
   docker compose up -d --build
   ```
3. **工程化特性**：通过数据卷挂载（Volume）实现了 SQLite 会话历史、HuggingFace 向量模型缓存、知识库文件的持久化。容器销毁重启不丢失业务数据，开箱即用，彻底解决多环境部署的依赖一致性问题。

## 🎯 效果演示

1. **多工具并发调用**：输入 北京、上海、厦门天气，Agent 通过线程池在同一毫秒内并发请求 3 个城市接口，并汇总回答。
2. **两阶段检索与降噪**：输入特定硬件型号，Agent 先混合检索召回 Top10，再通过 Rerank 精排剔除无关废话，最终精准输出。
3. **思考过程可视化**：前端可展开查看完整的“思考 - 调用 - 观察 - 回答”链路。
```

