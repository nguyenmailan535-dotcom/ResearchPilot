# ResearchPilot Architecture

ResearchPilot is an evidence-grounded technical research agent organized around AgentRunner,
provider abstraction, ToolRegistry, memory consolidation, and MCP support. It is implemented as a
cohesive set of native tools instead of introducing LangChain or LangGraph.

## Product scenario

A user provides local papers, design documents, or source code and asks a complex question such
as: "Compare Redis Streams and RabbitMQ for durable Agent task execution and recommend one for
this project." The Agent decomposes the request, retrieves evidence for each sub-question,
reflects on missing information, verifies exact chunks, writes a cited report, and stores stable
project decisions for future sessions.

This differs from one-shot RAG: retrieval is a tool selected repeatedly by the Agent, and the
workflow includes evidence-gap reflection, citation validation, a cited Evidence Ledger, and an
auditable event log.

## Architecture

```text
Web UI / CLI / FastAPI
       |
Persistent task state + resumable SSE events
       |
AgentRunner (ReAct loop)
       |
       +-- research_delegate -> 2-3 parallel read-only evidence researchers
       |                         +-- research_sources / search / read only
       |                         +-- structured findings + citation validation
       |                         +-- main-Agent synthesis and writes
       +-- research_ingest  -> Unstructured parsing + page/section semantic chunking
       +-- research_search  -> BM25 / BGE-M3 / RRF / metadata filter / rewrite / web fallback
       +-- research_read    -> exact evidence lookup
       +-- research_decision-> cited, versioned Evidence Ledger
       +-- research_report  -> citation + Claim/Evidence verification
       +-- research_sources -> corpus and audit statistics
                +-------------+----------------+
                |                              |
        Milvus retrieval plane          SQLite control plane
        dense + sparse indexes          tasks / events / ledger
```

SQLite remains the source of truth for sources, evidence chunks, the Evidence Ledger, asynchronous
task state and replayable events. Milvus is a rebuildable retrieval projection: Unstructured
element metadata and chunks are synchronized incrementally, BGE-M3 emits normalized 1024-dimensional
dense vectors, a Milvus BM25 Function generates sparse vectors, and HNSW serves dense ANN search.
The default semantic chunker treats Unstructured page and section boundaries as hard evidence
boundaries, then packs adjacent elements into bounded, sentence-aware windows. `research_search`
returns Top-5 evidence by default and can filter source IDs, page ranges, sections and element
types before either sparse or dense recall. When the caller explicitly permits external evidence,
an insufficient local result after confidence-based Query Rewrite is routed to the configured web
provider and clearly marked as non-RF evidence.
The local SQLite/NumPy backend remains available for unit tests and degraded development runs.
For a non-trivial corpus, run `researchpilot research sync-index --workspace <workspace>` after ingest so
embedding and index construction happen before the first interactive query. The Docker profile uses
Milvus standalone with embedded etcd and local persistent storage, avoiding an external object-store
dependency for the single-node resume/demo deployment.
Generic conversation memory remains owned by the ResearchPilot runtime. Citation IDs are stable
(`RF-<source>-<chunk>`) across unchanged re-indexing.

For substantial questions with independent dimensions, the main Agent can delegate two or three
evidence investigations concurrently. Delegated researchers have a deliberately read-only tool
surface: they can inspect sources, search, and read exact chunks, but cannot ingest documents,
save reports, execute shell commands, or mutate memory/the Evidence Ledger. Each worker returns
structured findings, gaps, queries and RF citations; the coordinator validates citation IDs,
resolves conflicts, performs final synthesis and remains the only writer.

## Design references

ResearchPilot was implemented for this repository and does not copy the following projects. The
star counts below are a 2026-09-25 snapshot; its workflow is informed by their publicly documented
ideas:

- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) (~29.6k stars):
  planner/executor research decomposition, parallel evidence gathering, and source-tracked report
  generation.
- [Local Deep Researcher](https://github.com/langchain-ai/local-deep-researcher) (~9.4k stars):
  iterative reflection on knowledge gaps followed by refined searches.
- [PaperQA2](https://github.com/Future-House/paper-qa) (~9.2k stars): agentic search, evidence
  gathering, grounded synthesis, and in-text citations.
- [Khoj](https://github.com/khoj-ai/khoj) (~37.5k stars): self-hosted personal knowledge and
  long-lived assistant context.

See each referenced project's license before reusing any of its code. Their repositories are design
references only and are not vendored into ResearchPilot. Third-party dependencies and inherited
MIT-licensed code retain their original notices in `LICENSE` and dependency metadata.

## Quick start

```powershell
# Index a folder of local PDFs and Markdown notes.
researchpilot research ingest .\papers --workspace .\demo-workspace

# Inspect deterministic retrieval without spending model tokens.
researchpilot research search "消息重复消费 幂等" --workspace .\demo-workspace `
  --strategy hybrid --no-answer-threshold 0.75

# Run the ResearchPilot Agent over the same workspace.
researchpilot agent --workspace .\demo-workspace

# Start the API and browser UI at http://127.0.0.1:18791
researchpilot serve --host 127.0.0.1 --port 18791 --workspace .\demo-workspace
```

Then ask:

```text
Use the research-flow skill. Compare the indexed approaches to message delivery reliability,
identify disagreements, and save a cited recommendation for project agent-runtime.
```

Validate a generated report independently:

```powershell
researchpilot research verify .\demo-workspace\research\reports\message-queues.md `
  --workspace .\demo-workspace
```

Run a reproducible retrieval evaluation:

```jsonl
{"id":"q1","query":"How are duplicate deliveries handled?","relevant_citations":["RF-a1b2c3d4-2"]}
```

```powershell
researchpilot research evaluate .\eval.jsonl --workspace .\demo-workspace --top-k 5 `
  --output .\results\bm25.json
```

Compare BM25, multilingual dense retrieval, weighted hybrid RRF, and deterministic query
reflection on the checked-in 40-question benchmark:

```powershell
pip install -e ".[research,api,eval]"

researchpilot research benchmark `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\final.json
```

See the [benchmark protocol, raw results, and limitations](../benchmarks/researchflow/README.md).

Select a chunk configuration on the development set and calibrate abstention with positive and
negative questions:

```powershell
researchpilot research chunk-ablation `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo

researchpilot research calibrate-threshold `
  .\benchmarks\researchflow\cardinality_abstention_dev_20.jsonl `
  --workspace .\research-demo --strategy hybrid
```

Run the independent grounded-answer evaluation:

```powershell
researchpilot research e2e-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\e2e-final.json

# Offline Judge over saved answers (does not regenerate the answers).
researchpilot research judge-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  .\benchmarks\researchflow\results\e2e-p0-p1-final.json `
  --workspace .\research-demo --judge-model deepseek-v4-pro `
  --output .\benchmarks\researchflow\results\judge-deepseek-v4-pro.json

# Optional, more expensive Claim—Evidence semantic verification.
researchpilot research judge-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  .\benchmarks\researchflow\results\e2e-p0-p1-final.json `
  --workspace .\research-demo --judge-model deepseek-v4-pro `
  --semantic-entailment --limit 1

# RAGAS is isolated because its OpenAI/LangChain pins conflict with the online runtime.
docker compose -f docker-compose.research.yml build researchflow
docker compose -f docker-compose.research.yml --profile eval build researchflow-eval
docker compose -f docker-compose.research.yml --profile eval run --rm researchflow-eval `
  /data/workspace/research/evaluations/e2e-bge-m3-milvus-resumed.json `
  --workspace /data/workspace --model deepseek-v4-pro `
  --output /data/workspace/research/evaluations/ragas-bge-m3.json

# If a provider/model failure interrupted only part of the metrics, reuse successful
# per-metric scores and rerun only the missing metrics. Use --limit for a smoke test.
docker compose -f docker-compose.research.yml --profile eval run --rm researchflow-eval `
  /data/workspace/research/evaluations/e2e-bge-m3-milvus-resumed.json `
  --workspace /data/workspace --model deepseek-v4-pro `
  --resume-from /data/workspace/research/evaluations/ragas-bge-m3.json `
  --output /data/workspace/research/evaluations/ragas-bge-m3.json
```

## What is deliberately out of scope

- A Feishu-specific channel. Existing delivery channels remain compatible, but they are delivery
  adapters rather than the project's core contribution.
- An online LLM judge in the serving hot path. The optional judge is evaluation-only; deterministic
  retrieval, citation, Claim/Evidence, latency and token metrics remain the primary signals.

## Evaluation

The repository includes a 40-question bilingual retrieval development set and a separate
16-question end-to-end holdout set. Raw per-case outputs are checked in. Evaluation measures:

- Recall@K and MRR for evidence retrieval
- citation validity and paragraph citation coverage
- task completion rate
- average/P50/P95 latency, tool calls, prompt/completion tokens, and explicit price-based cost
- unanswerable recall, false-answer rate, Claim/Evidence support, and Bad Case taxonomy
- changes after hybrid retrieval and deterministic query reflection

Keep the dataset, evaluator, and raw JSON results in the repository so every resume number is
reproducible.

The checked-in MiniLM/NumPy benchmark is a migration baseline. Current BGE-M3/Milvus raw outputs
are checked in separately; mapped citation labels still require human audit before the migration
metrics are treated as final ground truth.
