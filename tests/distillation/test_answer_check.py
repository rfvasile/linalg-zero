import pytest

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase


@pytest.mark.parametrize(
    "ground_truth, final_answer, expected",
    [
        ("-2.0", "-2.0", True),
        ("-2.0", "3.5", False),
        # parse_string returns None for non-numeric strings → verify_answers must reject.
        ("-2.0", "not a number", False),
    ],
)
def test_check_final_answers(
    base: MultiTurnWithToolUseBase, ground_truth: str, final_answer: str, expected: bool
) -> None:
    inputs = [{"ground_truth": ground_truth}]
    result = base.check_final_answers(final_answers=[final_answer], inputs=inputs)
    assert result == [expected]
