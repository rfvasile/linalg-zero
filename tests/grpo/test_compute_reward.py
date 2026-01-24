import asyncio

import pytest

from linalg_zero.grpo.envs.linear_algebra.compute_reward import answer_correct, think_correct, validate_answer
from linalg_zero.grpo.general_rm import calculate_reward
from linalg_zero.grpo.types import RunConfig, SolveResult

ANSWER = "<think>reason</think><answer>-2.0</answer>"


@pytest.mark.parametrize(
    "ground_truth, completion, expected",
    [
        (-2.0, ANSWER, True),  # parsed answer matches ground truth
        (-2.0, "<think>r</think><answer>5.0</answer>", False),  # wrong value
        (-2.0, "<think>r</think>no answer tag", False),  # no <answer>
        (-2.0, "<think>r</think><answer>not a number</answer>", False),  # unparseable
    ],
)
def test_validate_answer(ground_truth: float, completion: str, expected: bool) -> None:
    assert validate_answer(ground_truth, completion) is expected


@pytest.mark.parametrize(
    "completion, has_think, has_answer",
    [
        (ANSWER, True, True),
        ("<answer>1</answer>", False, True),
        ("<think>r</think>", True, False),
        ("plain", False, False),
    ],
)
def test_think_and_answer_correct(completion: str, has_think: bool, has_answer: bool) -> None:
    assert think_correct(completion) is has_think
    assert answer_correct(completion) is has_answer


def _config() -> RunConfig:
    return RunConfig(project_id="p", project="proj", model_provider="mp", dataset_path="d")


def _result(*, reward: float, forced_stop: bool) -> SolveResult:
    return SolveResult(reward=reward, messages=[], info={"forced_stop": forced_stop})


def test_calculate_reward_forced_stop_is_minus_one() -> None:
    reward, label = asyncio.run(calculate_reward(_result(reward=1.0, forced_stop=True), _config()))
    assert reward == -1
    assert label == "Max token trajectory"


def test_calculate_reward_real_passes_env_reward_through() -> None:
    reward, label = asyncio.run(calculate_reward(_result(reward=0.75, forced_stop=False), _config()))
    assert reward == 0.75
    assert label == "real_reward"


def test_calculate_reward_unknown_type_raises() -> None:
    config = _config()
    config.reward_type = "general_rm"  # removed path — must now be rejected
    with pytest.raises(ValueError, match="Invalid reward type"):
        asyncio.run(calculate_reward(_result(reward=0.0, forced_stop=False), config))
