"""Bounded parallel research delegation for evidence-grounded workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class ResearchDelegateTool(Tool):
    """Run independent, read-only evidence investigations in parallel."""

    def __init__(self, manager: "SubagentManager") -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "research_delegate"

    @property
    def description(self) -> str:
        return (
            "Delegate 2-3 independent research subquestions to read-only subagents in parallel. "
            "Each subagent can list sources, search evidence, and read exact RF citations, but "
            "cannot ingest documents, save reports, or modify the Evidence Ledger. The result "
            "contains validated citations for the main agent to synthesize and verify."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 60,
                            },
                            "question": {"type": "string", "minLength": 5},
                        },
                        "required": ["label", "question"],
                    },
                    "description": "Independent evidence questions that can run concurrently.",
                },
                "max_concurrency": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Maximum delegated researchers running at once.",
                },
            },
            "required": ["tasks"],
        }

    async def execute(
        self,
        tasks: list[dict[str, str]],
        max_concurrency: int = 3,
    ) -> str:
        result = await self._manager.run_research_tasks(
            tasks,
            max_concurrency=max_concurrency,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
