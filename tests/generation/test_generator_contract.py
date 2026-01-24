import json

import pytest

from linalg_zero.generator.models import Question, Task, Topic
from linalg_zero.generator.registry import FactoryRegistry, create_default_registry
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.lib import get_lib


def _all_registered_tasks() -> list[Task]:
    return create_default_registry().list_problem_types(Topic.LINEAR_ALGEBRA)


@pytest.mark.parametrize("task", _all_registered_tasks(), ids=lambda t: t.name)
def test_factory_produces_round_trippable_question(default_registry: FactoryRegistry, task: Task) -> None:
    factory = default_registry.get_factory(Topic.LINEAR_ALGEBRA, task)
    question = factory()

    _assert_question_shape(question, task)
    _assert_stepwise_round_trips(question)


def _assert_question_shape(question: Question, task: Task) -> None:
    assert isinstance(question, Question)
    assert question.is_valid, f"{task.name}: factory produced an invalid question"
    assert question.problem_type == task
    assert question.topic == Topic.LINEAR_ALGEBRA
    assert question.question.strip(), f"{task.name}: empty question text"
    assert question.answer.strip(), f"{task.name}: empty answer text"

    assert question.tool_calls_required == question.difficulty.value, (
        f"{task.name}: tool_calls_required={question.tool_calls_required} "
        f"does not match difficulty={question.difficulty.value}"
    )
    assert len(question.stepwise) == question.tool_calls_required, (
        f"{task.name}: stepwise has {len(question.stepwise)} entries, expected {question.tool_calls_required}"
    )

    # Composite Question.answer is a bundle of per-step results; golden carries the single final value.
    assert "final_answer" in question.golden, f"{task.name}: golden answer missing"
    parsed = parse_string(question.golden["final_answer"])
    assert parsed is not None, f"{task.name}: golden final_answer {question.golden['final_answer']!r} did not parse"


def _assert_stepwise_round_trips(question: Question) -> None:
    """Each recorded step must re-derive its result by invoking the real lib function."""
    lib = get_lib()
    for step in question.stepwise:
        tool_name = step["tool"]
        recorded_result = parse_string(step["result"])
        assert recorded_result is not None, f"{question.problem_type.name}: step {step['step_id']} has no result"

        input_kwargs = json.loads(step["verification"]["input"])
        actual_result = lib[tool_name](**input_kwargs)

        assert verify_answers(recorded_result, actual_result), (
            f"{question.problem_type.name}: step {step['step_id']} ({tool_name}) "
            f"recorded result {recorded_result} does not match lib result {actual_result}"
        )
