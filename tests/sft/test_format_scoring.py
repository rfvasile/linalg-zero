import pytest

from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback

PREFIX = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _context(turn_content: str) -> list[dict]:
    return [*PREFIX, {"role": "assistant", "content": turn_content}]


@pytest.mark.parametrize(
    "turn, expected",
    [
        ("<think>x</think><tool_call>{}</tool_call>", 3.0),  # think + tool pair
        ("<think>x</think><answer>y</answer>", 3.0),  # think + answer fallback (no tool tags)
        ("plain text, no tags", 0.0),  # nothing rewarded
        ("<think>x</think><think>y</think><tool_call>{}</tool_call>", 2.0),  # duplicate think → think reward lost
    ],
)
def test_partial_match_scores_tags_per_turn(callback: ToolCallingAccuracyCallback, turn: str, expected: float) -> None:
    assert callback.calculate_partial_match(_context(turn)) == expected, f"unexpected score for {turn!r}"


def test_partial_match_empty_conversation_is_zero(callback: ToolCallingAccuracyCallback) -> None:
    assert callback.calculate_partial_match(PREFIX) == 0


@pytest.mark.parametrize(
    "turn, expected",
    [
        ('<think>x</think><tool_call>{"name":"determinant","arguments":{}}</tool_call>', 1.0),
        ("<think>x</think><answer>y</answer>", 1.0),
        ("plain text, no tags", 0.0),
    ],
)
def test_exact_match_is_one_only_for_valid_format(
    callback: ToolCallingAccuracyCallback, turn: str, expected: float
) -> None:
    assert callback.calculate_exact_match(_context(turn)) == expected, f"unexpected score for {turn!r}"


def test_exact_match_empty_conversation_is_zero(callback: ToolCallingAccuracyCallback) -> None:
    assert callback.calculate_exact_match(PREFIX) == 0
