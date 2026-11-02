"""List-valued assistant content (Mistral reasoning chunks) through the real OpenAI client."""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.decorators import tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

pytestmark = [pytest.mark.asyncio, pytest.mark.allow_call_model_methods]

# Shapes documented by Mistral for reasoning models: the message content is a list of a
# "thinking" chunk (itself a list of text chunks) and "text" chunks. Streamed deltas carry
# lists while thinking, one list with the closing thinking chunk and the first text chunk,
# then plain strings.
_THINKING = {"type": "thinking", "thinking": [{"type": "text", "text": "Add 2 and 3."}]}
_STREAM_DELTAS: list[Any] = [
    [{"type": "thinking", "thinking": [{"type": "text", "text": "Add 2 "}]}],
    [{"type": "thinking", "thinking": [{"type": "text", "text": "and 3."}]}],
    [{"type": "thinking", "thinking": [], "closed": True}, {"type": "text", "text": "The sum"}],
    " is 5.",
]


def _completion(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": "synthetic",
        "object": "chat.completion",
        "created": 0,
        "model": "mistral-medium-3-5",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }


def _sse(deltas: list[dict[str, Any]], finish_reason: str) -> bytes:
    choices: list[dict[str, Any]] = [
        {"index": 0, "delta": delta, "finish_reason": None} for delta in deltas
    ]
    choices.append({"index": 0, "delta": {}, "finish_reason": finish_reason})
    payloads = [
        {
            "id": "synthetic",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mistral-medium-3-5",
            "choices": [choice],
        }
        for choice in choices
    ]
    return "".join(f"data: {json.dumps(p)}\n\n" for p in payloads).encode()


@pytest.mark.parametrize("stream", [False, True])
async def test_mistral_reasoning_content_chunks_run_tool_loop(stream: bool) -> None:
    requests: list[dict[str, Any]] = []
    calls: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 3})},
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        first = len(requests) == 1
        if not stream:
            message: dict[str, Any] = {"role": "assistant"}
            if first:
                message["content"] = [_THINKING]
                message["tool_calls"] = [tool_call]
            else:
                message["content"] = [_THINKING, {"type": "text", "text": "The sum is 5."}]
            return httpx2.Response(
                200, json=_completion(message, "tool_calls" if first else "stop")
            )
        if first:
            deltas: list[dict[str, Any]] = [
                {"role": "assistant", "content": [_THINKING]},
                {"tool_calls": [dict(tool_call, index=0)]},
            ]
        else:
            deltas = [{"content": content} for content in _STREAM_DELTAS]
        return httpx2.Response(
            200,
            content=_sse(deltas, "tool_calls" if first else "stop"),
            headers={"content-type": "text/event-stream"},
        )

    async with AsyncOpenAI(
        api_key="synthetic",
        base_url="https://api.mistral.ai/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler), trust_env=False),
    ) as client:
        agent = Agent(
            name="Mistral",
            tools=[add],
            model=OpenAIChatCompletionsModel("mistral-medium-3-5", client),
        )
        if stream:
            result = Runner.run_streamed(agent, "What is 2 + 3?")
            async for _ in result.stream_events():
                pass
        else:
            result = await Runner.run(agent, "What is 2 + 3?")  # type: ignore[assignment]

    assert result.final_output == "The sum is 5."
    assert calls == [(2, 3)]
    reasoning = [item.raw_item for item in result.new_items if item.type == "reasoning_item"]
    assert [r.summary[0].text for r in reasoning] == ["Add 2 and 3.", "Add 2 and 3."]
    # The replayed history carries the answer text only, as for string content.
    replayed = [m for m in requests[1]["messages"] if m["role"] == "assistant"]
    assert replayed[0]["tool_calls"][0]["id"] == "call_1"
    assert not replayed[0].get("content")
