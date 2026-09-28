"""Remap stable evaluation labels after a deliberate corpus rechunk.

This is a migration aid, not an automatic ground-truth generator. It matches each old evidence
chunk to the highest token-Jaccard candidate from the same source and page, records the score, and
keeps the original benchmark immutable so low-score mappings can be audited manually.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from nanobot.research.store import tokenize


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _chunks(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            "SELECT c.*, s.path, s.title FROM chunks c "
            "JOIN sources s ON s.id = c.source_id"
        )
    )


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def build_mapping(
    old_db: Path,
    new_db: Path,
    citations: set[str],
    *,
    minimum_score: float,
) -> dict[str, dict[str, Any]]:
    with _connect(old_db) as old_connection, _connect(new_db) as new_connection:
        old_by_id = {str(row["id"]).upper(): row for row in _chunks(old_connection)}
        new_by_source: dict[str, list[sqlite3.Row]] = {}
        for row in _chunks(new_connection):
            new_by_source.setdefault(Path(str(row["path"])).name.lower(), []).append(row)

        mapping: dict[str, dict[str, Any]] = {}
        for citation in sorted(citations):
            old = old_by_id.get(citation)
            if old is None:
                raise ValueError(f"Old citation is absent from source corpus: {citation}")
            source = Path(str(old["path"])).name.lower()
            source_candidates = new_by_source.get(source, [])
            same_page = [row for row in source_candidates if row["page"] == old["page"]]
            candidates = same_page or source_candidates
            if not candidates:
                raise ValueError(f"No new chunks found for source: {source}")
            score, best = max(
                ((_jaccard(str(old["content"]), str(row["content"])), row) for row in candidates),
                key=lambda item: (item[0], str(item[1]["id"])),
            )
            if score < minimum_score:
                raise ValueError(
                    f"Mapping score {score:.4f} is below {minimum_score:.4f}: {citation}"
                )
            mapping[citation] = {
                "new": str(best["id"]).upper(),
                "source": source,
                "old_page": old["page"],
                "new_page": best["page"],
                "score": round(score, 4),
            }
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-db", type=Path, required=True)
    parser.add_argument("--new-db", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--suffix", default="_bge_m3")
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()

    rows_by_dataset: dict[Path, list[dict[str, Any]]] = {}
    citations: set[str] = set()
    for dataset in args.datasets:
        rows = [
            json.loads(line)
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows_by_dataset[dataset] = rows
        citations.update(
            str(citation).upper()
            for row in rows
            for citation in row.get("relevant_citations", [])
        )

    mapping = build_mapping(
        args.old_db,
        args.new_db,
        citations,
        minimum_score=args.minimum_score,
    )
    for dataset, rows in rows_by_dataset.items():
        for row in rows:
            row["relevant_citations"] = [
                mapping[str(citation).upper()]["new"]
                for citation in row.get("relevant_citations", [])
            ]
        output = dataset.with_name(f"{dataset.stem}{args.suffix}{dataset.suffix}")
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{dataset} -> {output} ({len(rows)} cases)")

    scores = [float(item["score"]) for item in mapping.values()]
    report = {
        "old_db": str(args.old_db),
        "new_db": str(args.new_db),
        "mapping_method": "same-source/page token Jaccard",
        "requires_human_audit": True,
        "citations": len(mapping),
        "minimum_score": min(scores) if scores else None,
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mapping": mapping,
    }
    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "mapping"}))


if __name__ == "__main__":
    main()
