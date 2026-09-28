# ResearchFlow 简历与面试材料

## 简历项目名称

**ResearchFlow —— 轻量 Agentic RAG 论文研究 Agent**

技术栈：Python、ReAct、SubAgent、FastAPI、Milvus、Unstructured、BGE-M3、BM25、HNSW、RRF、Cross-Encoder、SQLite、SSE、RAGAS

## 简历项目描述（推荐版）

- 实现 `ContextBuilder → AgentRunner → ToolRegistry → Tool Execution` 核心运行链路，基于
  ReAct 实现 LLM 推理、Tool 选择、Observation 回填与多轮迭代的 Agent Loop，并通过迭代上限、
  超时及异常回传控制执行边界。
- 将论文检索封装为 Tool 接入 Agent Loop；主 Agent 将 2–3 个独立研究问题委派给只读 SubAgent
  并行检索，限制其仅可查看来源、检索与读取证据；子任务返回结构化结论、证据缺口和 Citation
  ID，由主 Agent 统一校验引用、解决冲突、综合报告并写入 Evidence Ledger。
- 构建论文知识库检索链路，基于 Unstructured 保留标题、正文元素、页码与章节 metadata；使用
  BGE-M3 生成 Dense 向量，通过 Milvus HNSW 与内置 BM25 Sparse Index 完成双路召回，经加权
  RRF 融合排序，并引入可选
  Cross-Encoder Rerank 提升 Top-K 证据相关性。
- 设计可追溯的 Evidence Grounding 机制，为 Chunk 生成 Citation ID，支持回答从引用回溯至原文；
  实现引用有效性与 Claim—Evidence 校验，降低论文分析中的无依据生成。
- 基于 FastAPI、SQLite 与 SSE 实现异步任务、并发控制、异常任务恢复标记与 Agent 执行轨迹推送；
  建立覆盖 Recall@K、Hit@K、MRR、Faithfulness、Answer Relevancy、引用有效率、延迟与 Token
  消耗的分层评测体系。40 条 BGE-M3/Milvus 迁移回归集上，Hybrid Recall@5 较 BM25 提升
  54.1%、MRR 提升 20.6%；16 条端到端测试任务完成率与引用有效率均为 100%，平均/P95
  延迟为 33.20/105.89 秒，平均输入/输出 Token 为 19.18k/1.10k，并通过结构化 Judge 与
  RAGAS 对生成质量进行离线复核。

## 一句话版本

基于 nanobot Runtime 构建可审计论文研究 Agent，通过混合检索、稳定引用、异步任务与 SSE
实时轨迹完成端到端研究流程，并以 40 条检索集和 16 条独立问题验证召回与引用质量。

## 面试开场（约 40 秒）

ResearchFlow 是我在读完 nanobot v0.1.4 源码后完成的二次开发项目。原版 Runtime 已经提供了
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
- RAGAS Answer Relevancy 已完成 16 条（0.8928）；Faithfulness 因模型服务余额不足仍待补跑，
  简历不得写成“RAGAS 全量评测完成”。
- 端到端生成受模型版本和随机性影响，简历数字以仓库保存的
  `e2e-bge-m3-milvus.json` 为准。

