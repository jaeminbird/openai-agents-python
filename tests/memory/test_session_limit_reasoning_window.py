from __future__ import annotations

from typing import Any

import pytest

from agents import Agent, RunConfig, Runner, SessionSettings, SQLiteSession, TResponseInputItem
from agents.testing import ScriptedModel, assistant_message

# History written by a reasoning model: each model response starts with its reasoning item.
REASONING_HISTORY: list[TResponseInputItem] = [
    {"role": "user", "content": "q1"},
    {"type": "reasoning", "id": "rs_1", "summary": []},
    {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
        "status": "completed",
    },
    {"type": "function_call_output", "call_id": "call_1", "output": "found"},
    {"type": "reasoning", "id": "rs_2", "summary": []},
    {
        "type": "message",
        "id": "msg_2",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "a1", "annotations": []}],
    },
    {"role": "user", "content": "q2"},
    {"type": "reasoning", "id": "rs_3", "summary": []},
    {
        "type": "message",
        "id": "msg_3",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "a2", "annotations": []}],
    },
]

_STORED_OUTPUT_TYPES = {"message", "function_call"}


def _outputs_missing_reasoning(items: list[Any]) -> list[str]:
    """Return stored output item IDs that are not preceded by their response's reasoning item.

    The Responses API rejects these with "Item '...' of type '...' was provided without its
    required 'reasoning' item".
    """
    missing: list[str] = []
    response_has_reasoning = False
    for item in items:
        item_type = item.get("type")
        if item_type == "reasoning":
            response_has_reasoning = True
        elif item.get("role") == "user" or item_type == "function_call_output":
            # The next model response starts after an input item.
            response_has_reasoning = False
        elif item_type in _STORED_OUTPUT_TYPES and item.get("id") and not response_has_reasoning:
            missing.append(item["id"])
    return missing


def _labels(items: list[Any]) -> list[str]:
    return [item.get("id") or item.get("call_id") or item.get("content") for item in items]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    ("limit", "expected_history"),
    [
        # Window starts at the function call, after its reasoning item.
        (7, ["rs_2", "msg_2", "q2", "rs_3", "msg_3"]),
        # Window starts at the tool output.
        (6, ["rs_2", "msg_2", "q2", "rs_3", "msg_3"]),
        # Window starts at the assistant message, after its reasoning item.
        (4, ["q2", "rs_3", "msg_3"]),
        # Window already starts at an input item: unchanged.
        (3, ["q2", "rs_3", "msg_3"]),
        (5, ["rs_2", "msg_2", "q2", "rs_3", "msg_3"]),
    ],
)
async def test_session_limit_window_never_starts_inside_a_reasoning_response(
    tmp_path, limit: int, expected_history: list[str], streamed: bool
) -> None:
    session = SQLiteSession("limit_reasoning", tmp_path / "s.db")
    await session.add_items(REASONING_HISTORY)
    model = ScriptedModel([[assistant_message("ok")]])
    agent = Agent(name="assistant", model=model)
    run_config = RunConfig(session_settings=SessionSettings(limit=limit))

    if streamed:
        result = Runner.run_streamed(agent, "q3", session=session, run_config=run_config)
        async for _ in result.stream_events():
            pass
    else:
        await Runner.run(agent, "q3", session=session, run_config=run_config)

    sent = list(model.calls[0].input)
    assert _outputs_missing_reasoning(sent) == []
    assert _labels(sent) == [*expected_history, "q3"]
    session.close()


@pytest.mark.asyncio
async def test_session_limit_keeps_leading_items_without_response_ids(tmp_path) -> None:
    session = SQLiteSession("limit_plain", tmp_path / "s.db")
    await session.add_items(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
    )
    model = ScriptedModel([[assistant_message("ok")]])
    agent = Agent(name="assistant", model=model)

    await Runner.run(
        agent,
        "q3",
        session=session,
        run_config=RunConfig(session_settings=SessionSettings(limit=3)),
    )

    assert [item.get("content") for item in model.calls[0].input] == ["a1", "q2", "a2", "q3"]
    session.close()
