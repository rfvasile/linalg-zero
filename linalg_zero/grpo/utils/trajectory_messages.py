from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import art


def messages_and_choices_to_messages(messages_and_choices: art.MessagesAndChoices) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in messages_and_choices:
        if isinstance(item, dict):
            messages.append(cast(dict[str, Any], item))
            continue

        if hasattr(item, "message"):  # openai Choice-like
            choice = cast(Any, item)
            messages.append(choice.message.model_dump())
            continue

        messages.append(cast(dict[str, Any], item))
    return messages


def extract_tool_name_sequence(traj: art.Trajectory) -> tuple[str, ...]:
    if not traj.messages_and_choices:
        return ()
    messages = messages_and_choices_to_messages(traj.messages_and_choices)
    tool_names: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tool_call in tool_calls:
            fn = tool_call.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)
    return tuple(tool_names)


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_messages: list[dict[str, Any]] = []
    for msg in messages:
        cleaned_messages.append({k: v for k, v in msg.items() if v is not None})
    return cleaned_messages
