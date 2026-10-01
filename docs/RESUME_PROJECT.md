# ResearchPilot 简历与面试材料

## 简历项目名称

**ResearchPilot｜专业文献 Agentic RAG 助手**

技术栈：Python、Agent Harness、FastAPI、Milvus、Unstructured、BGE-M3、BM25、RRF、RAGAS、Tavily

## 简历项目描述（推荐版）

- **上下文管理：**实现 System Prompt、会话历史、Skills、Memory、Tool Schema 与用户请求的上下文组装，并通过 Tool Result 截断与 Skills 渐进披露机制降低 Context Window 占用。
- **记忆系统：**实现 HISTORY.md + MEMORY.md 分层记忆，通过 Memory Consolidation 对历史会话进行摘要压缩，并支持 Agent 主动沉淀长期记忆，实现跨会话信息复用。
- **文档解析与向量索引：**使用 Unstructured 解析 PDF 并进行语义切分；基于 BGE-M3 生成 Dense Embedding，将 Chunk、页码及 Metadata 持久化至 Milvus，并采用 HNSW 构建向量索引。
- **混合检索：**构建 BM25 + Dense Retrieval 双路召回，通过 RRF 融合并结合 Metadata Filter 返回 Top-5 Evidence；低相关结果触发 Query Rewrite 与二次检索，仍未命中时降级至 Web Search。
- **效果评测：**构建40条检索回归集与16条端到端问答集；相比BM25基线，Hybrid Retrieval的Recall@5由0.4625提升至0.7125，MRR@5由0.3858提升至0.4654；RAGAS Faithfulness和Answer Relevancy分别达到0.8773和0.9134。

## 一句话版本

基于 nanobot Runtime 构建可审计论文研究 Agent，通过混合检索、稳定引用、异步任务与 SSE
实时轨迹完成端到端研究流程，并以 40 条检索集和 16 条独立问题验证召回与引用质量。

## 面试开场（约 40 秒）

ResearchPilot 是我在读完 nanobot v0.1.4 源码后完成的二次开发项目。原版 Runtime 已经提供了
Agent 循环、工具注册和会话能力，我主要解决的是研究场景中的三个工程问题：第一，回答必须能
回溯到本地论文原文；第二，长任务需要异步执行并向前端实时展示；第三，检索优化不能只凭主观
体验，需要有可复现评测。因此我实现了混合检索和稳定引用体系、SQLite 异步任务与 SSE 事件流，
并建立了检索、拒答、引用蕴含和端到端多层评测。项目最终可以通过 Docker Compose 一键启动。

## 可深入讲解的技术点

### 1. 为什么不只使用向量检索

BM25 对英文论文中的精确术语表现较强，但中文问题到英文论文的跨语言召回较弱；多语言向量检索
可以补足语义匹配，但首位精确度不稳定。因此系统使用加权 RRF 融合两路排名。40 条评测中，
BM25 Recall@5 为 0.4625，BGE-M3 Dense 为 0.7000，调参后的混合检索达到 0.7125。

### 2. 如何保证引用可验证

摄取阶段将每个证据块分配稳定的 `RF-<source>-<chunk>` 标识；生成前先读取精确证据块，生成后
扫描报告中的引用并检查其是否存在，同时计算段落引用覆盖率。前端点击引用会调用精确证据 API，
展示论文标题、页码和原始文本。离线评测进一步按 Claim 绑定原始 Evidence，先检查数字、实体与
词项覆盖，再可选调用独立 Judge 输出结构化蕴含标签；引用存在不再等同于引用支持结论。

### 3. 为什么使用 SQLite

项目是面向个人研究的单机 Agent，SQLite 能同时持久化语料元数据、任务状态、事件和 Evidence Ledger，
部署成本低。通过 WAL 模式和最大并发限制控制写入竞争；服务启动时将中断的 pending/running
任务标记为 failed，避免任务永远停留在运行状态。

### 4. 查询反思是否一定更好

不是。当前规则式反思的 Recall@5 为 0.6875，低于调参后 Hybrid 的 0.7125，P95 则从
994.27 ms 增加到 2149.80 ms。因此它只作为低置信度查询的降级路径，不能宣称全面优于 Hybrid。

### 5. 为什么 Reranker 没有默认开启

迁移前的 Cross-Encoder 消融将 Recall@5 提升到 0.7125、MRR 提升到 0.6071，但本机 CPU
P95 达 13.52 秒，
相比 Hybrid 的约 134 ms 代价过高，因此只作为显式或低置信度二阶段能力。这体现的是效果、
延迟和部署成本的权衡，而不是简单堆模型。

## 边界与诚实表述

- 项目是基于 nanobot Runtime 的二次开发，不应描述成从零实现完整 Agent 框架。
- 原 V1/V2 数字属于 MiniLM/NumPy 迁移前基线；简历中的 Recall/MRR 使用新保存的
  `retrieval-bge-m3-milvus-v1-tuned.json`，不得把两套结果混用。
- `768/120` 重建后的引用标签由旧数据机械映射，须完成人工抽检后才能把迁移评测写成最终指标；
  当前 0.7125/0.4654 明确标注为 BGE-M3/Milvus 迁移回归集结果。
- 默认端到端评分采用确定性规则；LLM Judge 仅在配置独立 Judge 模型时作为辅助指标。
- Judge 使用 V4-Pro 非思考模式与 temperature=0；思考模式会忽略温度，不适合宣称确定性评测。
- 100% 引用有效率不等同于所有表述均事实正确，规则式 Claim 支持率也需要人工复核。
- RAGAS 已完成全部 16 条（Faithfulness 0.8773、Answer Relevancy 0.9134）；使用 V4-Pro
  非思考模式、按指标断点续评，原始结果保存在 `ragas-deepseek-v4-pro-bge-m3.json`。
- 端到端生成受模型版本和随机性影响，简历数字以仓库保存的
  `e2e-bge-m3-milvus.json` 为准。

