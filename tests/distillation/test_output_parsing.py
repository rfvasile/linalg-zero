import pytest

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase

TOOL_RESPONSE_CTX = [{"role": "tool", "content": "[[1, 2], [3, 4]]"}]

WELL_FORMED_TOOL_CALL_JSON = (
    '{"thought":"compute it","tool_call":{"name":"determinant","arguments":{"matrix":[[1,2],[3,4]]}},'
    '"final_answer":null,"completed":false}'
)
WELL_FORMED_ANSWER_JSON = '{"thought":"done","tool_call":null,"final_answer":"-2.0","completed":true}'
BOTH_PRESENT_JSON = (
    '{"thought":"x","tool_call":{"name":"determinant","arguments":{}},"final_answer":"-2.0","completed":true}'
)
NEITHER_JSON = '{"thought":"x","tool_call":null,"final_answer":null,"completed":false}'


@pytest.mark.parametrize(
    "message, context, expected_ok, expected_reason_substring",
    [
        ("not json", [], False, "malformed JSON"),
        (NEITHER_JSON, [], False, "missing tool_call or final_answer"),
        (WELL_FORMED_ANSWER_JSON, [], False, "answer without tool response"),
        (BOTH_PRESENT_JSON, TOOL_RESPONSE_CTX, False, "both tool_call and final_answer present"),
        (WELL_FORMED_TOOL_CALL_JSON, [], True, "ok"),
        (WELL_FORMED_ANSWER_JSON, TOOL_RESPONSE_CTX, True, "ok"),
    ],
)
def test_extract_structured_output_policy(
    base: MultiTurnWithToolUseBase,
    message: str,
    context: list[dict],
    expected_ok: bool,
    expected_reason_substring: str,
) -> None:
    parsed, reason = base.extract_structured_output(message, context=context)
    assert (parsed is not None) == expected_ok
    assert expected_reason_substring in reason


# Unstructured outputs come from non-JSON LLMs. Test the strict_format gate and answer policy.
WELL_FORMED_TOOL_CALL_TEXT = (
    '<think>compute it</think><tool_call>{"name":"determinant","arguments":{"matrix":[[1,2],[3,4]]}}</tool_call>'
)
WELL_FORMED_ANSWER_TEXT = "<think>done</think><answer>-2.0</answer>"
MISSING_THINK_TEXT = '<tool_call>{"name":"determinant","arguments":{}}</tool_call>'


@pytest.mark.parametrize(
    "message, context, expect_completed, expect_tool_call",
    [
        (WELL_FORMED_TOOL_CALL_TEXT, [], False, True),
        (WELL_FORMED_ANSWER_TEXT, TOOL_RESPONSE_CTX, True, False),
    ],
)
def test_extract_non_structured_output_accepts_well_formed(
    base: MultiTurnWithToolUseBase,
    message: str,
    context: list[dict],
    expect_completed: bool,
    expect_tool_call: bool,
) -> None:
    parsed = base.extract_non_structured_output(message, context=context)
    assert parsed is not None
    assert parsed.completed is expect_completed
    assert (parsed.tool_call is not None) is expect_tool_call


def test_extract_non_structured_output_rejects_missing_think(base: MultiTurnWithToolUseBase) -> None:
    # strict_format=True (the default) requires the <think>…</think> prefix.
    assert base.extract_non_structured_output(MISSING_THINK_TEXT, context=[]) is None
