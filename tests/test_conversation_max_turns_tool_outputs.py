"""A run that stops at max_turns must not leave tool calls without outputs in its conversation."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from agents import Agent, MaxTurnsExceeded, RunErrorHandlerResult, Runner, function_tool
from agents.models.openai_responses import OpenAIResponsesModel

pytestmark = [pytest.mark.asyncio, pytest.mark.allow_call_model_methods]


class _FakeConversations:
    """Stores conversation items and rejects a response request the way the API does when a
    function call in the conversation has no output."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.count = 0

    def _missing_output(self, items: list[dict[str, Any]]) -> str | None:
        calls = [i["call_id"] for i in items if i.get("type") == "function_call"]
        outputs = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
        return next((call_id for call_id in calls if call_id not in outputs), None)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/conversations/conv_1/items"):
            self.items += body["items"]
            self.created += body["items"]
            return httpx2.Response(200, json={"object": "list", "data": body["items"]})
        assert body["conversation"] == "conv_1"
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [{"type": "message", "role": "user", "content": inputs}]
        missing = self._missing_output(self.items + inputs)
        if missing:
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "message": f"No tool output found for function call {missing}.",
                        "type": "invalid_request_error",
                        "param": "input",
                        "code": None,
                    }
                },
            )
        self.count += 1
        if self.count == 1:
            output: list[dict[str, Any]] = [
                {
                    "type": "function_call",
                    "id": f"fc_{n}",
                    "call_id": f"call_{n}",
                    "name": "weather",
                    "arguments": json.dumps({"city": city}),
                    "status": "completed",
                }
                for n, city in ((1, "Paris"), (2, "Tokyo"))
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": f"msg_{self.count}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ]
        self.items += [*inputs, *output]
        response = {
            "id": f"resp_{self.count}",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        if not body.get("stream"):
            return httpx2.Response(200, json=response)
        events: list[dict[str, Any]] = [
            {"type": "response.created", "response": dict(response, output=[])},
            *(
                {"type": "response.output_item.done", "output_index": i, "item": item}
                for i, item in enumerate(output)
            ),
            {"type": "response.completed", "response": response},
        ]
        data = "".join(
            f"event: {e['type']}\ndata: {json.dumps(dict(e, sequence_number=n))}\n\n"
            for n, e in enumerate(events)
        )
        return httpx2.Response(
            200, content=data.encode(), headers={"content-type": "text/event-stream"}
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("handled", [False, True])
async def test_max_turns_in_conversation_keeps_conversation_usable(
    stream: bool, handled: bool
) -> None:
    @function_tool
    def weather(city: str) -> str:
        return f"sunny in {city}"

    server = _FakeConversations()
    client = AsyncOpenAI(
        api_key="synthetic",
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(server.handler), trust_env=False
        ),
        max_retries=0,
    )
    agent = Agent(name="a", model=OpenAIResponsesModel("gpt-5", client), tools=[weather])
    kwargs: dict[str, Any] = {"conversation_id": "conv_1", "max_turns": 1}
    if handled:
        kwargs["error_handlers"] = {
            "max_turns": lambda _data: RunErrorHandlerResult(final_output="stopped")
        }

    async def run(text: str, **kw: Any) -> Any:
        if stream:
            result = Runner.run_streamed(agent, text, **kw)
            async for _ in result.stream_events():
                pass
            return result
        return await Runner.run(agent, text, **kw)

    if handled:
        assert (await run("weather?", **kwargs)).final_output == "stopped"
    else:
        with pytest.raises(MaxTurnsExceeded):
            await run("weather?", **kwargs)

    # The tools ran, so their outputs belong in the conversation.
    assert [(i["call_id"], i["output"]) for i in server.created] == [
        ("call_1", "sunny in Paris"),
        ("call_2", "sunny in Tokyo"),
    ]
    # The next user turn in the same conversation still works.
    result = await run("hi", conversation_id="conv_1")
    assert result.final_output == "hello"
