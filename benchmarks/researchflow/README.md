# ResearchFlow 检索评测

本目录保存两套可复现的领域检索评测，用于比较 ResearchFlow 的 BM25、向量检索、
混合检索、查询反思和可选 Reranker。V1 是 6 篇论文、480 个稳定引用块的原始快照；
当前语料已扩展为 10 篇论文。新的 Unstructured 元素打包 + `768/120` 固定窗口索引包含
1,243 个稳定引用块；
旧 `research-demo` 快照仍保留，便于复现历史指标。

## 数据集

- 40 条人工编写并标注的真实技术问题：24 条英文、16 条中文。
- 难度分布：7 条 easy、14 条 medium、19 条 hard。
- 主题分布：HyperLogLog 7 条、DREX 11 条、ULL 10 条、SeiveSketch 10 条、
  Mycroft 1 条、Astral 1 条。
- 每条问题使用一个或多个 `RF-<source>-<chunk>` 稳定引用作为相关证据标签。
- 原始评测集位于
  [`cardinality_sketch_40.jsonl`](./cardinality_sketch_40.jsonl)，完整逐题结果位于
  [`results/final.json`](./results/final.json)。

这里的 Recall@K 是严格的“标注证据块召回率”，不是答案正确率。如果另一个未标注
块也能回答问题，仍会按未命中处理。因此该指标更适合比较不同检索方案，而不能直接
解释为端到端 Agent 的回答准确率。

## 对比方案

1. **BM25**：ResearchStore 内置的中英文分词与 BM25 检索。
2. **向量检索**：生产链路使用 `BAAI/bge-m3` 生成 1024 维归一化 Dense 向量，并由
   Milvus HNSW 索引召回；早期 MiniLM/NumPy 指标仅保留作迁移前历史基线。
3. **混合检索**：Milvus BM25 Sparse 与 BGE-M3 Dense 双路召回；在 V2 开发集上网格搜索后，
   使用 BM25:Dense=`1:1`、RRF rank constant=`20`，不在 V1 回归集上调参。
4. **查询反思**：对中英文领域术语进行确定性扩展，执行第二轮混合检索，再融合两轮
   排名。此版本不调用 LLM，避免评测结果受模型随机性和 API 状态影响。

## V2 扩展语料诊断

新增 RAP、RVCC、SMB 和 SpreadSketch 后，建立了 20 条中英双语检索问题：每篇论文
4 条单论文问题，另含 4 条跨论文比较问题。它是扩展语料的开发/诊断集，不与 V1 指标
拼接，也不宣称为不可调参的最终测试集。原始数据与逐题输出分别见
[`cardinality_retrieval_v2_20.jsonl`](./cardinality_retrieval_v2_20.jsonl) 和
[`results/retrieval-v2-20-reranked.json`](./results/retrieval-v2-20-reranked.json)。

| 方法 | Recall@1 | Recall@3 | Recall@5 | Success@5 | MRR@5 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | **0.5250** | **0.6250** | 0.6500 | 0.7000 | 0.6017 | 158.01 ms |
| 向量检索 | 0.4250 | 0.4750 | 0.5750 | 0.6000 | 0.4917 | 112.25 ms |
| Hybrid RRF | 0.4750 | **0.6250** | 0.6500 | 0.7000 | 0.5792 | 151.76 ms |
| Reflected Hybrid | 0.4750 | 0.5750 | 0.6500 | 0.7000 | 0.5750 | 334.73 ms |
| Hybrid + Reranker | **0.6500** | **0.7250** | **0.8000** | **0.8500** | **0.7225** | 10.75 s |

这组结果说明扩展语料中的精确英文术语仍有利于 BM25；当前向量模型并未稳定提升首位排序。
Reranker 将 Recall@5 从 0.65 提高到 0.80，但 CPU P95 增至 10.75 秒，因此继续只作为
显式二阶段能力。另建的 5 正/5 负无答案小样本仅得到 0.60 平衡准确率，说明现有单一
融合分数在困难负例上区分度不足；结果见
[`results/abstention-v2-10.json`](./results/abstention-v2-10.json)，不使用它替换 V1 的
0.75 阈值。

## 最终结果

在同一台本地机器、向量索引已缓存的条件下运行。耗时是 40 条查询的单次墙钟时间，
仅用于观察相对开销，不是严谨的吞吐基准。

| 方法 | Recall@1 | Recall@3 | Recall@5 | Success@5 | MRR@5 | 40 条耗时 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.4250 | 0.5375 | 0.5375 | 0.5500 | 0.4958 | 3.390 s |
| 向量检索 | 0.3250 | 0.5625 | 0.6125 | 0.6250 | 0.4487 | 0.441 s |
| 混合检索（RRF） | **0.5000** | **0.6125** | 0.6500 | 0.6500 | 0.5729 | 4.482 s |
| 查询反思 + 混合检索 | **0.5000** | 0.5875 | **0.6625** | **0.6750** | **0.5746** | 9.522 s |

新增逐查询计时后，本机缓存索引的 P95 分别为：BM25 95.46 ms、向量 14.00 ms、
Hybrid 134.15 ms、Reflected Hybrid 283.09 ms。单次本地运行只适合比较相对开销，
不应解释为生产 SLA；原始输出见 [`results/p0-p1-final.json`](./results/p0-p1-final.json)。

可选 Cross-Encoder Reranker 将 Hybrid Recall@5 从 0.6500 提升到 0.7125、MRR 从
0.5729 提升到 0.6071，但本机 CPU P95 从约 134 ms 增加到 13.52 s。因此它保留为
显式 `--rerank`/低置信度二阶段能力，不作为默认在线路径。完整消融见
[`results/reranker-ablation.json`](./results/reranker-ablation.json)。

相对纯 BM25，普通混合检索的 Recall@5 提升 **0.1125（20.9%）**，MRR@5 提升
**0.0771（15.5%）**。查询反思将 Recall@5 再提升 0.0125、MRR@5 提升 0.0017，
但耗时约为普通混合检索的 2.12 倍。因此它只适合作为低置信度查询的第二阶段，
不应笼统宣称“查询反思全面优于混合检索”。

## 分层结果（Recall@5）

| 方法 | 英文（24） | 中文（16） | Easy（7） | Medium（14） | Hard（19） |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.8125 | 0.1250 | 0.5000 | 0.5000 | 0.5789 |
| 向量检索 | 0.7500 | 0.3750 | 0.4286 | 0.4286 | 0.7895 |
| 混合检索（RRF） | 0.7917 | **0.4375** | 0.4286 | **0.5714** | **0.7895** |
| 查询反思 + 混合检索 | **0.8125** | **0.4375** | **0.5000** | **0.5714** | **0.7895** |

BM25 在英文原词匹配上很强，但中文问题对英文论文的跨语言召回只有 0.1250；向量
检索把该值提高到 0.3750，混合检索进一步提高到 0.4375。这是本实验中采用多语言
语义检索的主要收益。

## 消融记录

仓库保留了两次诊断实验，便于解释优化来自哪里，而不是只保留最好结果。

| 实验版本 | 向量 R@5 | 混合 R@5 | 反思 R@5 |
|---|---:|---:|---:|
| 整块向量 + 等权 RRF + 标题伪反馈 | 0.2750 | 0.5000 | 0.4750 |
| 分段向量 + BM25 2:1 RRF + 标题伪反馈 | 0.6000 | 0.6500 | 0.6000 |
| 分段向量 + 加权 RRF + 确定性查询扩展 | 0.6000 | 0.6500 | **0.6625** |

把约 1,200 字符的证据块直接编码改为细粒度段落编码，是向量召回提升的主要来源；
删除基于首轮标题的伪相关反馈，则避免了查询漂移。

## 复现

```powershell
pip install -e ".[research]"

nanobot research benchmark `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\final.json
```

首次执行会下载嵌入模型并建立本地索引，后续运行读取缓存。要重建向量，添加
`--rebuild-vectors`。

## 无答案与切块消融

`cardinality_abstention_dev_20.jsonl` 包含 10 条可回答问题和 10 条语料外问题，用于选择
无答案阈值，不能作为最终测试集。阈值以可回答准确率和不可回答召回率的平衡准确率选择：

```powershell
nanobot research calibrate-threshold `
  .\benchmarks\researchflow\cardinality_abstention_dev_20.jsonl `
  --workspace .\research-demo --strategy hybrid

nanobot research chunk-ablation `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo
```

切块消融比较 `256/64`、`384/64`、`480/80`、`768/120` 以及结构感知 `480/80`，
选择规则固定为 Recall@K 优先、MRR 次之、P95 延迟再次之。最终 holdout 不参与选择。

本次开发集实验推荐固定切块 `768/120`：映射后 Recall@5 为 0.9500、MRR 为 0.8042、
P95 检索延迟为 135.72 ms。这里的相关性通过原标注块的 source/page 与内容重叠映射到
新切块，因此不能和上文严格 Citation-ID Recall 直接比较；完整结果见
[`results/chunk-ablation.json`](./results/chunk-ablation.json)。在重建正式语料和重新标注
holdout 前，现有 1,200/160 索引仍保留，以避免让已发布的稳定引用失效。

## BGE-M3 + Milvus 正式重建结果（10 篇论文）

根据开发集消融结论，在独立工作区用 Unstructured 提取页级元素，再将相邻元素按页和章节
打包并执行固定窗口 `768/120`，完整重建 10 篇论文，得到 1,243 个证据块。块长均值为
668.5 字符、中位数 706、P95 765，仅 22 块短于 200 字符。Milvus 中的精确行数为 1,243，
二次同步 `upserted=0/deleted=0`，验证了索引同步幂等性。

旧标签依据 source、page 和 token Jaccard 机械映射到新引用；46 个唯一引用的平均/最低
映射分数为 0.6197/0.4041。因此下列结果属于可复现的迁移回归测试，作为正式 ground truth
发布前仍需人工抽检。映射记录见
[`results/citation-remap-bge-m3-packed.json`](./results/citation-remap-bge-m3-packed.json)。

V2 开发集用于选择 RRF 参数，最优配置为 BM25:Dense=`1:1`、`k=20`。固定参数后在 40 条
V1 映射回归集复测：

| 方法 | Recall@5 | MRR@5 | P95 |
|---|---:|---:|---:|
| BM25 | 0.4625 | 0.3858 | 100.21 ms |
| BGE-M3 Dense | 0.7000 | 0.4196 | 852.47 ms |
| Hybrid RRF | **0.7125** | **0.4654** | 994.27 ms |
| Reflected Hybrid | 0.6875 | 0.4508 | 2149.80 ms |

Hybrid 相比 BM25 的 Recall@5 相对提升 **54.1%**、MRR@5 相对提升 **20.6%**。规则反思
在这次复测中效果和延迟都弱于已调参 Hybrid，因此只保留为低置信度降级策略，不作为默认链路。
原始结果见
[`results/retrieval-bge-m3-milvus-v1-tuned.json`](./results/retrieval-bge-m3-milvus-v1-tuned.json)，
调参轨迹见 [`results/rrf-tuning-bge-m3-v2.json`](./results/rrf-tuning-bge-m3-v2.json)。

20 条拒答开发集推荐阈值为 0.75：平衡准确率 0.85、可回答准确率 0.70、不可回答召回率
1.00、误答率 0；它只用于选择阈值，不作为最终泛化成绩。原始结果见
[`results/abstention-bge-m3-milvus-dev.json`](./results/abstention-bge-m3-milvus-dev.json)。

16 条端到端复测结果为：任务完成率 1.0000、答案通过率 0.8750、概念覆盖率 0.8958、
引用有效率 1.0000、相关引用召回率 0.7292、段落引用覆盖率 0.8281、规则 Claim 支持率
0.7776，平均/P95 延迟 33.20/105.89 秒，平均输入/输出 Token 为 19,183.56/1,095.94。
DeepSeek V4-Pro 结构化 Judge 平均总分为 1.0000；RAGAS 的 16 条 Answer Relevancy 为
0.8928，Faithfulness 因模型服务余额不足未完成，不能与已完成指标混写。原始结果分别见
[`results/e2e-bge-m3-milvus.json`](./results/e2e-bge-m3-milvus.json)、
[`results/judge-deepseek-v4-pro-bge-m3.json`](./results/judge-deepseek-v4-pro-bge-m3.json) 和
[`results/ragas-deepseek-v4-pro-bge-m3.partial.json`](./results/ragas-deepseek-v4-pro-bge-m3.partial.json)。

## 历史 MiniLM/NumPy 端到端基线

为避免只证明“检索到了证据”，另建了 16 条未参与检索方案调参的中英双语问题，检查
最终回答中的必需概念、引用有效性、相关证据召回、段落引用覆盖、延迟和 token。
数据集为 [`cardinality_e2e_holdout_16.jsonl`](./cardinality_e2e_holdout_16.jsonl)，逐题
原始回答和指标保存在 [`results/e2e-final.json`](./results/e2e-final.json)。

| 指标 | 结果 |
|---|---:|
| 任务完成率 | 1.0000 |
| 答案通过率 | 0.9375 |
| 必需概念覆盖率 | 0.8594 |
| 引用有效率 | 1.0000 |
| 相关证据引用召回率 | 0.8229 |
| 段落引用覆盖率 | 0.8344 |
| Claim—Evidence 规则支持率 | 0.7833 |
| 平均 / P95 延迟 | 11.1692 / 16.9055 s |
| Prompt Tokens（平均 / 总计） | 18,344.75 / 293,516 |
| Completion Tokens（平均 / 总计） | 994.25 / 15,908 |

```powershell
nanobot research e2e-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\e2e-final.json
```

该测试使用当前配置的 `deepseek-flash`。美元成本没有填写，因为运行时未提供带日期的
价格参数；评测器会在传入每百万输入/输出 Token 单价后计算，未配置时返回 `null`，
不会把未知成本伪装成 0。93.75%“答案通过率”表示 15/16 条回答成功完成、至少覆盖一半标注概念
且引用了相关证据，并不表示所有自然语言陈述都达到同等比例的事实正确性。端到端结果
会受到模型服务版本和生成随机性的影响，表格记录的是仓库中
`e2e-p0-p1-final.json` 对应运行。

最新 P0/P1 结果保存于 [`results/e2e-p0-p1-final.json`](./results/e2e-p0-p1-final.json)，
Bad Case 自动归类为：1 条检索/排序未命中、9 条存在未引用段落、8 条被规则验证器标记为
至少一个不充分支持 Claim。这些标签用于建立回归集，仍需人工复核，不能直接当作事实错误率。

### 历史答案的 DeepSeek V4-Pro LLM-as-Judge

对上述 16 条已保存答案使用 `deepseek-v4-pro` 进行独立辅助评分。Judge 读取题目、必需概念、
答案、原始 Gold Evidence 和答案实际引用的原文块，按正确性、忠实度、完整性、引用对齐四个
0–2 分量表输出结构化结果。仅提供 Gold Chunk 会把合法的扩展引用误判为 unsupported，因此
两类证据必须分开传入。
为使 `temperature=0` 真正生效，评测显式使用 V4-Pro 的非思考模式；思考模式会忽略 temperature，
不适合作为要求重复性的固定量表 Judge。

| Judge 指标 | 结果 |
|---|---:|
| 平均总分（0–1） | 1.0000 |
| 正确性（0–2） | 2.0000 |
| 忠实度（0–2） | 2.0000 |
| 完整性（0–2） | 2.0000 |
| 引用对齐（0–2） | 2.0000 |
| 平均置信度 | 0.9738 |
| Prompt / Completion Tokens | 61,070 / 7,850 |

```powershell
nanobot research judge-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  .\benchmarks\researchflow\results\e2e-p0-p1-final.json `
  --workspace .\research-demo --judge-model deepseek-v4-pro `
  --output .\benchmarks\researchflow\results\judge-deepseek-v4-pro.json
```

逐题结果见 [`results/judge-deepseek-v4-pro.json`](./results/judge-deepseek-v4-pro.json)。
可选 `--semantic-entailment` 会再进行逐 Claim 语义验证，单条冒烟结果保存在
[`results/judge-deepseek-v4-pro-semantic-smoke.json`](./results/judge-deepseek-v4-pro-semantic-smoke.json)；
它的 Token 开销明显更高，默认不运行。美元成本因未固化带日期的单价而保持 `null`。
满分仅表示该 Judge 在给定证据和量表下未发现问题，不是人工事实正确率；四条原低分样本的
复核记录见 [`JUDGE_AUDIT.md`](./JUDGE_AUDIT.md)，人工签字仍待完成。

## 局限与下一步

- V1 指标是 6 篇论文语料快照上的结果；10 篇语料的 V2 仅是 20 条开发/诊断集成绩，二者不能
  混用，也不能外推到开放域检索。
- 标签由一人标注，且部分问题可能存在未标出的等价证据块。
- 40 条检索数据同时用于诊断检索方案；16 条端到端问题是独立测试，但仍由一人标注。
- 当前“反思”是可复现的规则式术语扩展，不等同于 LLM 对证据缺口的推理。
- LLM Judge 仅作为可选辅助指标；需要抽样人工复核，不能取代引用、检索和
  Claim—Evidence 的确定性指标。

下一版应对等价证据进行多人复核，并增加人工事实忠实度、置信区间和 P95 延迟评测。
