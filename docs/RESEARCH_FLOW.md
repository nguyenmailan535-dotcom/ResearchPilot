# ResearchFlow

ResearchFlow is an evidence-grounded technical research agent built on nanobot's existing
AgentRunner, provider abstraction, ToolRegistry, memory consolidation, and MCP support. It is
implemented as a cohesive set of native tools instead of introducing LangChain or LangGraph.

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
Web UI / CLI / HTTP API
       |
Persistent task state + resumable SSE events
       |
nanobot AgentRunner (existing ReAct loop)
       |
       +-- research_ingest  -> fixed/structure-aware configurable chunking
       +-- research_search  -> BM25 / dense / RRF / reflection / optional reranking
       +-- research_read    -> exact evidence lookup
       +-- research_decision-> cited, versioned Evidence Ledger
       +-- research_report  -> citation + Claim/Evidence verification
       +-- research_sources -> corpus and audit statistics
                              |
                        SQLite research.db
```

The SQLite store contains sources, evidence chunks, an Evidence Ledger, and append-only research
events. Generic conversation memory remains owned by the nanobot runtime. The store also persists
API task state and replayable task events. Citation IDs are stable
(`RF-<source>-<chunk>`) across unchanged re-indexing.

## Design references

ResearchFlow was implemented for this repository and does not copy the following projects. The
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

See each upstream project's license before reusing any of its code. ResearchFlow currently uses
only the nanobot repository's existing MIT-licensed runtime and original implementation code in
this branch.

## Quick start

```powershell
# Index a folder of local PDFs and Markdown notes.
nanobot research ingest .\papers --workspace .\demo-workspace

# Inspect deterministic retrieval without spending model tokens.
nanobot research search "消息重复消费 幂等" --workspace .\demo-workspace `
  --strategy hybrid --no-answer-threshold 0.75

# Run the normal nanobot Agent over the same workspace.
nanobot agent --workspace .\demo-workspace

# Start the API and browser UI at http://127.0.0.1:18791
nanobot serve --host 127.0.0.1 --port 18791 --workspace .\demo-workspace
```

Then ask:

```text
Use the research-flow skill. Compare the indexed approaches to message delivery reliability,
identify disagreements, and save a cited recommendation for project agent-runtime.
```

Validate a generated report independently:

```powershell
nanobot research verify .\demo-workspace\research\reports\message-queues.md `
  --workspace .\demo-workspace
```

Run a reproducible retrieval evaluation:

```jsonl
{"id":"q1","query":"How are duplicate deliveries handled?","relevant_citations":["RF-a1b2c3d4-2"]}
```

```powershell
nanobot research evaluate .\eval.jsonl --workspace .\demo-workspace --top-k 5 `
  --output .\results\bm25.json
```

Compare BM25, multilingual dense retrieval, weighted hybrid RRF, and deterministic query
reflection on the checked-in 40-question benchmark:

```powershell
pip install -e ".[research]"

nanobot research benchmark `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\final.json
```

See the [benchmark protocol, raw results, and limitations](../benchmarks/researchflow/README.md).

Select a chunk configuration on the development set and calibrate abstention with positive and
negative questions:

```powershell
nanobot research chunk-ablation `
  .\benchmarks\researchflow\cardinality_sketch_40.jsonl `
  --workspace .\research-demo

nanobot research calibrate-threshold `
  .\benchmarks\researchflow\cardinality_abstention_dev_20.jsonl `
  --workspace .\research-demo --strategy hybrid
```

Run the independent grounded-answer evaluation:

```powershell
nanobot research e2e-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  --workspace .\research-demo `
  --output .\benchmarks\researchflow\results\e2e-final.json

# Offline Judge over saved answers (does not regenerate the answers).
nanobot research judge-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  .\benchmarks\researchflow\results\e2e-p0-p1-final.json `
  --workspace .\research-demo --judge-model deepseek-v4-pro `
  --output .\benchmarks\researchflow\results\judge-deepseek-v4-pro.json

# Optional, more expensive Claim—Evidence semantic verification.
nanobot research judge-evaluate `
  .\benchmarks\researchflow\cardinality_e2e_holdout_16.jsonl `
  .\benchmarks\researchflow\results\e2e-p0-p1-final.json `
  --workspace .\research-demo --judge-model deepseek-v4-pro `
  --semantic-entailment --limit 1
```

## What is deliberately out of scope

- A Feishu-specific channel. Existing nanobot channels remain compatible, but they are delivery
  adapters rather than the project's core contribution.
- An external vector database. The benchmark layer uses a local FastEmbed/ONNX index with a
  corpus-addressed NumPy cache, keeping the project self-contained and reproducible.
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
