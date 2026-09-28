from types import SimpleNamespace

import pytest

from nanobot.research.ragas_eval import evaluate_ragas_results
from nanobot.research.store import ResearchStore


class _Metric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class _FailingMetric:
    async def ascore(self, **kwargs):
        raise RuntimeError("judge output was truncated")


@pytest.mark.asyncio
async def test_ragas_adapter_uses_answer_citations_as_context(tmp_path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("HyperLogLog estimates distinct cardinality.", encoding="utf-8")
    store = ResearchStore(tmp_path)
    indexed = store.ingest_file(source)
    citation = f"RF-{indexed['source_id'][:8]}-1"
    faithfulness = _Metric(0.8)
    relevancy = _Metric(0.9)

    result = await evaluate_ragas_results(
        store,
        [{"id": "q1", "query": "What does HLL estimate?", "answer": f"Cardinality [{citation}]."}],
        faithfulness_metric=faithfulness,
        relevancy_metric=relevancy,
    )

    assert result["faithfulness"] == 0.8
    assert result["answer_relevancy"] == 0.9
    assert result["completed_cases"] == 1
    assert result["failed_cases"] == 0
    assert result["details"][0]["contexts"] == 1
    assert "HyperLogLog" in faithfulness.calls[0]["retrieved_contexts"][0]


@pytest.mark.asyncio
async def test_ragas_adapter_records_per_metric_failure(tmp_path) -> None:
    store = ResearchStore(tmp_path)
    result = await evaluate_ragas_results(
        store,
        [{"id": "q1", "query": "question", "answer": "answer"}],
        faithfulness_metric=_FailingMetric(),
        relevancy_metric=_Metric(0.7),
    )

    assert result["completed_cases"] == 0
    assert result["failed_cases"] == 1
    assert result["faithfulness"] is None
    assert result["answer_relevancy"] == 0.7
    assert result["details"][0]["completed"] is False
    assert "RuntimeError" in result["details"][0]["errors"]["faithfulness"]
