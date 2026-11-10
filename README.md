
# 🤖 多模型研讨 Agent（Multi-Model Debate + Hybrid RAG + LangGraph）

基于 **LangGraph + FastAPI + SQLite + FAISS** 从零搭建的企业级多模型研讨智能体。项目完整实现了**多模型研讨（专家独立作答 → 交叉研讨 → 首席总结）**、ReAct 闭环、**Query改写+两阶段混合检索（粗排+精排）**、**多工具线程池并发调度**、**全链路 Trace ID 追踪**、**SSE 双通道流式输出**、ChatGPT 同款多会话管理，并深度集成了**无状态多模型透传、联网搜索、接口鉴权与工具错误自愈机制**。支持 Docker 多阶段构建与一键容器化交付。

## ✨ 核心工程亮点

- 🗣️ **多模型研讨（Multi-Model Debate）**：同一问题由多个不同厂商的模型并行作答，再让它们互相阅读彼此观点做交叉研讨，最后由首席模型汇总成一份结论。支持任意增删主/副模型，三个阶段的产出按轮次分块独立渲染（不会互相覆盖），研讨会话与单模型会话一样落库、出现在左侧历史记录中，聊天框顶部实时显示本轮的主模型与副模型。
- 🧠 **意图前置路由与防死循环**：设计正则匹配的 Intent Router，对时间/计算等确定性场景进行后端兜底隔离；在 LLM 节点动态判断 `tool_choice`（首次 `required` 强控，二次 `auto` 恢复），彻底解决大模型“幻觉不调工具”与“工具无限死循环”的痛点。
- ⚡ **多工具线程池并发调度**：针对工具调用属于 IO 密集型任务的特点，在 `tool_node` 中引入 `ThreadPoolExecutor`。当大模型一次性发起多个 Function Calling 时，系统并发请求第三方接口，多任务总耗时缩短 60% 以上，并通过 Future 机制保证上下文顺序绝对保真。
- 🔍 **三阶段全链路 RAG 优化**：
  - **查询改写（Query Rewrite）**：检索前由大模型将用户口语化提问提炼为高密度搜索关键词，打通表达鸿沟。
  - **混合粗排**：基于 **BM25 + FAISS 双路召回 + RRF 融合算法**，解决专有名词召回率低的问题。
  - **精排降噪（Rerank）**：引入轻量级 **Cross-Encoder (BAAI/bge-reranker-base)** 对粗排 Top10 进行深度交互打分，最终提纯 Top3 上下文，防范大模型“中间迷失”。
- 📄 **文档问答（Word / PDF 直接读）**：前端一键上传 `.docx` / `.pdf` / `.txt`，后端提取正文后注入上下文，可直接对合同、手册、论文提问；**单模型会话与多模型研讨都支持**——研讨时文档会同时进入专家首轮作答、交叉研讨、首席总结三个阶段。Word 用标准库解析（docx 本质是 zip + XML，**不额外引入依赖**，表格按 `单元格 | 单元格` 还原），PDF 用 `pypdf` 逐页抽取并标注页码；单文件上限 15MB / 20 万字，超长自动截断并明确告知模型，避免它编造文档里没有的内容。
- 🔐 **无状态自定义大模型透传**：支持用户在前端输入自己的 API Key 切换不同大模型（通义千问/智谱/DeepSeek/豆包/GPT/Claude 等）。后端基于 `contextvars` 实现请求级上下文隔离，密钥随请求传入，用完即焚，服务端零存储；配合接口白名单防范 SSRF，全链路日志强制脱敏，安全性极高。
- 🌐 **联网搜索开关与兜底**：集成博查（Bocha）联网搜索 API。前端提供“强制联网”开关，开启后后端通过 Prompt 注入强控模型调用 `search_web` 工具；同时支持智能自主联网（如询问实时新闻时），保证最新资讯的获取。
- 🛡️ **安全防护与错误自愈**：新增 **API Key 接口鉴权**与基于 `slowapi` 的**单 IP 限流**；在工具节点设计了**精准前缀匹配的异常自愈机制**，当接口超时或参数错误时，回流 Error 日志引导大模型自主修正参数重试（最多2次），大幅提升任务成功率。
- 📊 **全链路 Trace ID 追踪**：基于 Python `contextvars` 与自定义 `logging.Formatter`，实现从接口入口、Agent 节点、并发工具到第三方库的全链路日志追踪（完美解决线程池上下文丢失问题），大幅提升线上排障效率。
- 💻 **极致的双通道 SSE 流式体验**：
  - 基于 `StreamingResponse` 与前端 `ReadableStream`，实现“先极速流式打印思考过程，再平滑打字机输出最终回答”的极致体验。
  - 引入 `AbortController` 支持前端一键“停止生成”，实时掐断后端流，节约 Token 消耗。
- 💾 **持久化与会话管理**：摒弃内存字典，引入 SQLAlchemy + SQLite（开启 WAL 模式支持高并发读写）实现对话状态持久化；配合三级记忆架构（短期+摘要+长期），实现服务重启及多进程环境下的跨会话状态无缝恢复。
- 🔒 **并发安全与防踩踏**：废弃全局状态字典，基于 LangGraph State 的 `input_len` 增量截取 `final_messages`，实现并发请求下执行链路（思考过程）的无状态提取，彻底解决上下文污染与数据错乱。
- 🏗️ **DDD 分层与配置化解耦**：采用标准领域分层架构（`api` / `agent` / `core` / `db` / `middleware` / `tools`）。核心参数全部抽离至 `.env` 配置；支持 Docker **多阶段构建**减小镜像体积，并利用数据卷挂载一键部署。

## 🛠️ 技术栈

- **核心框架**：FastAPI + Uvicorn + SQLAlchemy
- **Agent 编排**：LangGraph (StateGraph) + ReAct 范式 + ThreadPoolExecutor 并发
- **大语言模型**：Qwen-Plus / GLM-4 / DeepSeek / Kimi (OpenAI 兼容协议)
- **检索增强 (RAG)**：FAISS + BM25 + Jieba + HuggingFace Embedding + BAAI/bge-reranker 精排
- **联网搜索**：Bocha AI Web Search API
- **安全防护**：SlowAPI (限流) + 自定义 API Key 鉴权
- **数据库**：SQLite (WAL 模式持久化) + 内存缓存 (天气/高频工具)
- **前端交互**：原生 HTML/JS + SSE 流式渲染 + localStorage 多模型配置
- **部署运维**：Docker + Docker Compose

## 📁 项目结构

```text
multi-model-agent-debate/
├── app/
│   ├── api/          # 路由层：鉴权、意图拦截、流式/同步接口、多会话查询
│   ├── agent/        # 核心层：LangGraph 状态机、LLM 节点、并发 Tool 节点
│   ├── core/         # 基础层：全局配置、Trace 日志、文本清洗、LLM 工厂
│   ├── db/           # 数据层：SQLite 模型、CRUD、向量库 (FAISS+BM25+精排) 
│   ├── middleware/   # 中间件层：API Key 鉴权与 IP 限流器
│   └── tools/        # 工具层：天气、计算、时间、RAG检索、联网搜索
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

### 1.1 如何新增一个模型（以豆包为例）

所有内置模型都来自 `.env` 里的 `MODEL_REGISTRY`（一段单行 JSON），**加模型不需要改后端代码**：

```json
"doubao-seed-2-1-lite":{
  "base_url":"https://ark.cn-beijing.volces.com/api/v3",
  "api_key":"你的火山方舟 ARK_API_KEY",
  "vendor":"字节跳动",
  "label":"豆包 Seed 2.1 Lite",
  "model":"doubao-seed-2-1-lite-260915",
  "extra_body":{"thinking":{"type":"disabled"}}
}
```

把这段追加到 `MODEL_REGISTRY` 的 JSON 里即可（注意逗号），重启后下拉框、多模型研讨都会自动出现。

- `model` 字段可选，表示**真正发给厂商的模型名**：注册表里的 key 可以起一个干净好读的名字（如上例的 `doubao-seed-2-1-lite`），真实模型 ID 放在 `model` 里。
  不写这个字段时，key 会直接当作模型名发给厂商（qwen / glm / deepseek / kimi 都是这样）。
- 豆包的 `model` 也可以直接填模型 ID（如 `doubao-seed-2-1-lite-260915`），以方舟控制台为准。
- `extra_body` 可选，会原样追加到请求体里。豆包 Seed 2.1 系列默认开启**深度思考**，一次简单问答要 20~60 秒，經常直接超时；
  按上面的写法关掉之后，同样的请求 1~2 秒就返回，工具调用和流式输出都正常。
- 想让它默认被选中：把 `static/index.html` 里的 `DEFAULT_MODEL_ID` 改成你的模型 key。
- 想让它默认被选中：把 `static/index.html` 里的 `DEFAULT_MODEL_ID` 改成你的模型 key。

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
1. 点击去 **[Releases](https://github.com/3038816077/multi-model-agent-debate/releases/tag/v1.0.0)** 页面下载 `all-MiniLM-L6-v2.zip`。
2. 将压缩包解压，把里面的 `hub` 文件夹放到你的电脑缓存目录：
   - Windows: `C:\Users\你的用户名\.cache\huggingface\`
   - Mac/Linux: `~/.cache/huggingface/`
3. 打开 `.env` 文件，设置 `HF_HUB_OFFLINE=1`，重新启动服务即可实现秒级冷启动。

## 🎯 效果演示

1. **极致的流式体验**：输入问题，AI 会先极速打字输出“思考链路”，随后平滑打字输出最终回答；点击右侧方块按钮可实时中断生成。
2. **多会话智能管理**：左侧边栏自动提取首句作为会话标题，支持一键新建，上下文完全隔离且持久化。
3. **多工具并发调用**：输入 `北京、上海、厦门天气`，Agent 通过线程池在同一毫秒内并发请求 3 个城市接口并汇总。
4. **实时联网搜索**：勾选输入框上方“联网搜索”，询问“最近 AI 新闻”，Agent 会自动调用搜索工具抓取最新资讯并总结。
5. **多模型动态切换**：点击右上角“模型配置”，下拉选择厂商（如 DeepSeek）并填入 API Key，即可无缝切换底层大模型，且密钥仅存本地浏览器，用完即焚。
6. **三阶段检索与降噪**：输入“核心硬件型号”，Agent 依次触发 Query改写 -> 混合召回 -> Rerank精排，精准剔除废话并在回答末尾附带来源文件。
7. **全链路追踪**：所有终端日志均带有唯一的 `[Trace_ID]` 标签，方便一键排障。
```
