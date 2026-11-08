
# 🤖 多工具智能助手 Agent (Hybrid RAG + LangGraph)

基于 **LangGraph + FastAPI + SQLite + FAISS** 从零搭建的企业级多工具智能体。项目完整实现了 ReAct 闭环、**Query改写+两阶段混合检索（粗排+精排）**、**多工具线程池并发调度**、**全链路 Trace ID 追踪**、**SSE 双通道流式输出（流式思考+打字机）**、ChatGPT 同款多会话管理，并深度集成了**接口鉴权、限流防护与工具错误自愈机制**。支持 Docker 多阶段构建与一键容器化交付。

## ✨ 核心工程亮点

- 🧠 **意图前置路由与防死循环**：设计正则匹配的 Intent Router，对时间/计算等确定性场景进行后端兜底隔离；在 LLM 节点动态判断 `tool_choice`（首次 `required` 强控，二次 `auto` 恢复），彻底解决大模型“幻觉不调工具”与“工具无限死循环”的痛点。
- ⚡ **多工具线程池并发调度**：针对工具调用属于 IO 密集型任务的特点，在 `tool_node` 中引入 `ThreadPoolExecutor`。当大模型一次性发起多个 Function Calling 时，系统并发请求第三方接口，多任务总耗时缩短 60% 以上，并通过 Future 机制保证上下文顺序绝对保真。
- 🔍 **三阶段全链路 RAG 优化**：
  - **查询改写（Query Rewrite）**：检索前由大模型将用户口语化提问提炼为高密度搜索关键词，打通表达鸿沟。
  - **混合粗排**：基于 **BM25 + FAISS 双路召回 + RRF 融合算法**，解决专有名词召回率低的问题。
  - **精排降噪（Rerank）**：引入轻量级 **Cross-Encoder (BAAI/bge-reranker-base)** 对粗排 Top10 进行深度交互打分，最终提纯 Top3 上下文，防范大模型“中间迷失”。
- 🛡️ **安全防护与错误自愈**：新增 **API Key 接口鉴权**与基于 `slowapi` 的**单 IP 限流**；在工具节点设计了**精准前缀匹配的异常自愈机制**，当接口超时或参数错误时，回流 Error 日志引导大模型自主修正参数重试（最多2次），大幅提升任务成功率。
- 📊 **全链路 Trace ID 追踪**：基于 Python `contextvars` 与自定义 `logging.Formatter`，实现从接口入口、Agent 节点、并发工具到第三方库的全链路日志追踪（完美解决线程池上下文丢失问题），大幅提升线上排障效率。
- 💻 **极致的前端交互与多会话管理**：
  - **SSE 双通道流式输出**：基于 `StreamingResponse` 与前端 `ReadableStream`，实现“先极速流式打印思考过程，再平滑打字机输出最终回答”的极致体验。
  - **ChatGPT 同款侧边栏**：结合 SQLite 持久化与 `localStorage`，实现历史会话秒级切换与新建。
  - **智能摘要标题**：侧边栏自动提取用户首句提问的前15个字符作为会话摘要标题，支持过长省略与悬浮查看。
- 💾 **持久化与会话管理**：摒弃内存字典，引入 SQLAlchemy + SQLite（开启 WAL 模式支持高并发读写）实现对话状态持久化；配合三级记忆架构（短期+摘要+长期），实现服务重启及多进程环境下的跨会话状态无缝恢复。
- 🔒 **并发安全与防踩踏**：废弃全局状态字典，基于 LangGraph State 的 `input_len` 增量截取 `final_messages`，实现并发请求下执行链路（思考过程）的无状态提取，彻底解决上下文污染与数据错乱。
- 🏗️ **DDD 分层与配置化解耦**：采用标准领域分层架构（`api` / `agent` / `core` / `db` / `middleware` / `tools`）。核心参数（切分大小、召回数量、并发数等）全部抽离至 `.env` 配置；支持 Docker **多阶段构建**减小镜像体积，并利用数据卷挂载一键部署。

## 🛠️ 技术栈

- **核心框架**：FastAPI + Uvicorn + SQLAlchemy
- **Agent 编排**：LangGraph (StateGraph) + ReAct 范式 + ThreadPoolExecutor 并发
- **大语言模型**：通义千问 Qwen-Plus (OpenAI 兼容协议)
- **检索增强 (RAG)**：FAISS + BM25 + Jieba + HuggingFace Embedding + BAAI/bge-reranker 精排
- **安全防护**：SlowAPI (限流) + 自定义 API Key 鉴权
- **数据库**：SQLite (WAL 模式持久化) + 内存缓存 (天气/高频工具)
- **前端交互**：原生 HTML/JS + SSE 流式接收渲染 + 多会话管理
- **部署运维**：Docker + Docker Compose
- **可观测性**：自定义 Trace ID 贯穿全链路

## 📁 项目结构

```text
agent-demo/
├── app/
│   ├── api/          # 路由层：鉴权、意图拦截、流式/同步接口、多会话查询
│   ├── agent/        # 核心层：LangGraph 状态机、LLM 节点、并发 Tool 节点
│   ├── core/         # 基础层：全局配置、Trace 日志、文本清洗
│   ├── db/           # 数据层：SQLite 模型、CRUD、向量库 (FAISS+BM25+精排) 
│   ├── middleware/   # 中间件层：API Key 鉴权与 IP 限流器
│   └── tools/        # 工具层：天气、计算、时间、RAG 检索
├── knowledge/        # 本地知识库目录 (支持 .txt/.md/.pdf)
├── static/           # 前端交互页面 (支持多会话与流式打字机)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py            # 启动入口
```

## 🚀 快速启动

### 1. 配置环境变量
复制根目录的 `env.txt` 并重命名为 `.env`，填入你的大模型与天气 API Key。
*(注：可在 `.env` 中配置 `API_KEY` 开启鉴权；首次启动保持 `HF_HUB_OFFLINE=0` 联网下载模型)*

### 2. 方式一：Docker 部署（强烈推荐）
在项目根目录执行：
```bash
docker compose up -d --build
```

### 3. 方式二：本地 Python 运行
```bash
pip install -r requirements.txt
python run.py
```

### 4. 访问页面
- **前端页面**：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- **接口文档**：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ⚠️ 网络异常无法下载模型怎么办？（离线兜底方案）
由于国内网络访问 HuggingFace 官方受限，如果你在首次启动时卡在 `正在加载向量模型`，请直接使用我们提供的离线模型包：
1. 点击去 **[Releases](https://github.com/3038816077/multi-tool-agent-demo/releases/tag/v1.0.0)** 页面下载 `all-MiniLM-L6-v2.zip`。
2. 将压缩包解压，把里面的 `hub` 文件夹放到你的电脑缓存目录：
   - Windows: `C:\Users\你的用户名\.cache\huggingface\`
   - Mac/Linux: `~/.cache/huggingface/`
3. 打开 `.env` 文件，设置 `HF_HUB_OFFLINE=1`，重新启动服务即可实现秒级冷启动。

## 🎯 效果演示

1. **极致的流式体验**：输入问题，AI 会先极速打字输出“思考链路”，随后平滑打字输出最终回答。
2. **多会话智能管理**：左侧边栏自动提取首句作为会话标题，支持一键新建，上下文完全隔离且持久化。
3. **多工具并发调用**：输入 `北京、上海、厦门天气`，Agent 通过线程池在同一毫秒内并发请求 3 个城市接口并汇总。
4. **三阶段检索与降噪**：输入口语化问题，Agent 依次触发 Query改写 -> 混合召回 -> Rerank精排，精准剔除废话并输出。
5. **工具自愈**：输入不存在的城市，触发生态工具报错并回流大模型自愈重试，最终生成友好兜底提示。
6. **全链路追踪**：所有终端日志均带有唯一的 `[Trace_ID]` 标签，方便一键排障。
```


