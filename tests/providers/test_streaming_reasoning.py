from __future__ import annotations

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def test_parse_chunks_preserves_reasoning_content_for_tool_round_trip() -> None:
    chunks = [
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "inspect ", "content": None},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "sources",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "research_sources",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    response = OpenAICompatProvider._parse_chunks(chunks)

    assert response.reasoning_content == "inspect sources"
    assert response.tool_calls[0].name == "research_sources"
