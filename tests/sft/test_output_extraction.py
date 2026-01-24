import pytest

from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback

TOOL_CALL_MSG = (
    '<think>compute it</think><tool_call>{"name":"determinant","arguments":{"matrix":[[1,2],[3,4]]}}</tool_call>'
)
ANSWER_MSG = "<think>done</think><answer>-2.0</answer>"
TOOL_RESPONSE_CTX = [{"role": "tool", "content": "-2.0"}]


@pytest.mark.parametrize(
    "message, context, expected",
    [
        ("plain text, no tags", [], None),  # invalid format
        (ANSWER_MSG, [], None),  # answer without a preceding tool response
        (TOOL_CALL_MSG, [], {"name": "determinant", "answer": None, "completed": False}),
        (ANSWER_MSG, TOOL_RESPONSE_CTX, {"name": None, "answer": "-2.0", "completed": True}),
    ],
)
def test_extract_exact_match_policy(
    callback: ToolCallingAccuracyCallback, message: str, context: list[dict], expected: dict | None
) -> None:
    parsed = callback.extract_exact_match(message, context=context)
    if expected is None:
        assert parsed is None, f"expected rejection for {message!r}"
        return
    assert parsed is not None
    assert (parsed.tool_call.name if parsed.tool_call else None) == expected["name"]
    assert parsed.final_answer == expected["answer"]
    assert parsed.completed is expected["completed"]


def test_add_message_appends_assistant_string(callback: ToolCallingAccuracyCallback) -> None:
    context: list[dict] = []
    callback.add_message("assistant", context, "hello")
    assert context[-1] == {"role": "assistant", "content": "hello"}


def test_add_message_appends_unstructured_tool_result(callback: ToolCallingAccuracyCallback) -> None:
    context: list[dict] = []
    callback.add_message(
        "tool", context, {"execution_result": "-2.0", "function_name": "determinant"}, unstructured=True
    )
    assert context[-1] == {"role": "tool", "content": "-2.0", "name": "determinant"}


def test_add_message_rejects_invalid_role(callback: ToolCallingAccuracyCallback) -> None:
    with pytest.raises(ValueError, match="Invalid role"):
        callback.add_message("system", [], "x")
