import json

import pytest

from linalg_zero.generator.models import Question, Task, Topic
from linalg_zero.generator.registry import FactoryRegistry
from linalg_zero.grpo.verify import parse_string, verify_answers

COMPOSITE_REPRESENTATIVES = [
    Task.TWO_TRANSPOSE_DETERMINANT,
    Task.TWO_COFACTOR_RANK,
    Task.THREE_TRANSPOSE_COFACTOR_RANK,
    Task.THREE_TRANSPOSE_COFACTOR_FROBENIUS,
]


@pytest.fixture
def composite_question(default_registry: FactoryRegistry, request: pytest.FixtureRequest) -> Question:
    task: Task = request.param
    return default_registry.get_factory(Topic.LINEAR_ALGEBRA, task)()


@pytest.mark.parametrize("composite_question", COMPOSITE_REPRESENTATIVES, ids=lambda t: t.name, indirect=True)
def test_first_step_is_independent(composite_question: Question) -> None:
    first_step = composite_question.stepwise[0]
    assert "dependent_on" not in first_step["verification"], (
        f"{composite_question.problem_type.name}: first step claims a dependency but has no upstream step"
    )


@pytest.mark.parametrize("composite_question", COMPOSITE_REPRESENTATIVES, ids=lambda t: t.name, indirect=True)
def test_dependent_inputs_match_referenced_step_results(composite_question: Question) -> None:
    """Every input_* field with a dependent_on reference must equal the referenced step's result."""
    stepwise = composite_question.stepwise
    for i, step in enumerate(stepwise[1:], start=1):
        dependent_on = step["verification"].get("dependent_on")
        assert dependent_on is not None, (
            f"{composite_question.problem_type.name}: step {i} has no dependent_on but is not the first step"
        )

        for input_name, ref_index in dependent_on.items():
            assert 0 <= ref_index < i, f"step {i} references index {ref_index}, must be in [0, {i})"
            recorded_input = json.loads(step["verification"][input_name])
            referenced_result = parse_string(stepwise[ref_index]["result"])

            assert verify_answers(recorded_input, referenced_result), (
                f"{composite_question.problem_type.name}: step {i} {input_name}={recorded_input} "
                f"does not match step {ref_index} result {referenced_result}"
            )
