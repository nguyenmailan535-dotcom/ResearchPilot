from __future__ import annotations

from pathlib import Path

from nanobot.research.metrics import classify_bad_case, estimate_cost_usd, percentile
from nanobot.research.store import ResearchStore
from nanobot.research.verification import verify_claim_evidence


def test_claim_evidence_checks_numbers_and_support(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "paper.md"
    source.write_text("The measured relative error is 1.04 divided by the square root of m." * 3)
    store = ResearchStore(workspace)
    store.ingest_file(source)
    citation = store.search("relative error 1.04")[0]["citation"]

    supported = verify_claim_evidence(
        store, f"The relative error is 1.04 divided by the square root of m [{citation}]."
    )
    unsupported = verify_claim_evidence(
        store, f"The relative error is 9.99 divided by the square root of m [{citation}]."
    )

    assert supported["entailed"] == 1
    assert unsupported["unsupported"] == 1
    assert unsupported["details"][0]["missing_numbers"] == ["9.99"]


def test_observability_helpers() -> None:
    assert percentile([1, 2, 3, 100], 95) == 85.45
    assert estimate_cost_usd(1_000_000, 500_000, prompt_usd_per_million=1, completion_usd_per_million=2) == 2
    labels = classify_bad_case(
        {
            "expected_answerable": False,
            "abstained": False,
            "citation_validity": 1,
            "paragraph_citation_coverage": 1,
            "concept_coverage": 1,
            "entailment": {"unsupported": 0},
        }
    )
    assert labels == ["false_answer"]
