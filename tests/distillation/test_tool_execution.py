import pytest

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase
from linalg_zero.distillation.data import FunctionInvocationInfo, ThoughtSchema


def _tool_call(name: str, arguments: dict) -> ThoughtSchema:
    return ThoughtSchema(
        thought="",
        tool_call=FunctionInvocationInfo(name=name, arguments=arguments),
        final_answer=None,
        completed=False,
    )


@pytest.mark.parametrize(
    "name, arguments, expected_prefix",
    [
        ("determinant", {"matrix": [[1, 2], [3, 4]]}, "-2"),
        ("does_not_exist", {}, "ERROR: Function 'does_not_exist' not found"),
        ("determinant", {"matrix": [[1, 2, 3], [4, 5, 6]]}, "ERROR: ValueError"),
    ],
)
def test_execute_dispatch(base: MultiTurnWithToolUseBase, name: str, arguments: dict, expected_prefix: str) -> None:
    results, stats = base._execute(inputs=[_tool_call(name, arguments)], active_indices=[0])

    assert len(results) == 1
    assert results[0]["function_name"] == name
    assert results[0]["execution_result"].startswith(expected_prefix), (
        f"got {results[0]['execution_result']!r}, expected prefix {expected_prefix!r}"
    )
    assert stats == {0: name}


def test_execute_skips_input_with_no_tool_call(base: MultiTurnWithToolUseBase) -> None:
    # Upstream filtering should prevent this, but _execute must skip safely if it slips through.
    no_call = ThoughtSchema(thought="", tool_call=None, final_answer="42", completed=True)
    results, stats = base._execute(inputs=[no_call], active_indices=[0])
    assert results == []
    assert stats == {}
