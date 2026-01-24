import pytest

from linalg_zero.grpo.task_selection import (
    ShuffleBagSampler,
    ToolCallsMixtureSampler,
    _deterministic_counts_from_probs,
    get_task_indices,
)
from linalg_zero.grpo.types import Action, CurriculumConfig, Task


def _task(tool_calls: int) -> Task:
    return Task(
        user_id="u",
        actions=[Action(name="determinant", kwargs={}) for _ in range(tool_calls)],
        instruction="i",
        outputs=[],
    )


# Six tasks bucketed by tool-call count: {1:[0,1], 2:[2,3], 3:[4,5]}.
TASKS = [_task(c) for c in (1, 1, 2, 2, 3, 3)]


def _curriculum() -> CurriculumConfig:
    return CurriculumConfig(enabled=True, metric="tool_calls", initial_max_tool_calls=1, min_total_tasks=1)


@pytest.mark.parametrize(
    "probs, total, expected",
    [
        ([0.2, 0.3, 0.5], 10, [2, 3, 5]),  # exact proportional split
        ([0.2, 0.3, 0.5], 0, [0, 0, 0]),  # nothing to allocate
        ([0.0, 0.0, 0.0], 3, [1, 1, 1]),  # degenerate mass → uniform
    ],
)
def test_deterministic_counts_from_probs(probs: list[float], total: int, expected: list[int]) -> None:
    counts = _deterministic_counts_from_probs(probs=probs, total=total)
    assert counts == expected
    assert sum(counts) == total
    assert all(c >= 0 for c in counts)


def test_deterministic_counts_remainder_goes_to_largest_fraction() -> None:
    # expected = [0.5, 1.5, 1.0]; floors [0,1,1] leave remainder 1 → the largest fraction (idx 0).
    assert _deterministic_counts_from_probs(probs=[0.5, 1.5, 1.0], total=3) == [1, 1, 1]


def test_get_task_indices_curriculum_is_monotonic() -> None:
    def selected(difficulty: float) -> set[int]:
        return set(
            get_task_indices(
                task_ids=None,
                start_index=0,
                end_index=-1,
                tasks=TASKS,
                curriculum=_curriculum(),
                difficulty=difficulty,
            )
        )

    easy, mid, full = selected(0.0), selected(0.5), selected(1.0)
    assert easy <= mid <= full, "selection must grow monotonically with difficulty"
    assert full == set(range(len(TASKS))), "difficulty 1.0 exposes the full set"
    assert easy, "min_total_tasks keeps the easiest selection non-empty"


def test_get_task_indices_task_ids_win_and_full_range() -> None:
    assert get_task_indices(
        task_ids=[3, 1], start_index=0, end_index=-1, tasks=TASKS, curriculum=_curriculum(), difficulty=1.0
    ) == [3, 1]
    assert get_task_indices(task_ids=None, start_index=0, end_index=-1, tasks_length=6) == list(range(6))


def test_shuffle_bag_sampler_covers_pool_and_is_deterministic() -> None:
    batch = ShuffleBagSampler(seed=1).sample_batch(eligible=[0, 1, 2, 3], batch_size=4)
    assert sorted(batch) == [0, 1, 2, 3], "a full-pool batch covers every eligible index before repeating"
    again = ShuffleBagSampler(seed=1).sample_batch(eligible=[0, 1, 2, 3], batch_size=4)
    assert batch == again, "same seed → identical batch"


def test_mixture_sampler_is_deterministic() -> None:
    def batch() -> list[int]:
        sampler = ToolCallsMixtureSampler(tasks=TASKS, indices=list(range(6)), curriculum=_curriculum(), seed=2)
        return sampler.sample_batch(difficulty=0.5, batch_size=4)

    assert batch() == batch()
