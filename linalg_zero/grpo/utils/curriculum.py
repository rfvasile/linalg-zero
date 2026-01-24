from __future__ import annotations

import math
from collections import Counter
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from linalg_zero.grpo.task_selection import ShuffleBagSampler, ToolCallsMixtureSampler, get_task_indices
from linalg_zero.grpo.types import RunConfig

if TYPE_CHECKING:
    from art.utils.iterate_dataset import DatasetBatch

COVERAGE_LOG_MAX_TOOL_CALLS_BUCKET = 3
COVERAGE_PRINT_EVERY_STEPS = 50


class CurriculumCoverageTracker:
    """
    Per-step task coverage logging.

    Tracks which task indices have been sampled so far, broken down by a simple difficulty proxy:
    the number of teacher tool calls in the task (`len(task.actions)`).

    Intended for debugging curriculum exposure (e.g., ensuring harder tasks actually show up),
    not as a training signal.
    """

    def __init__(self, *, tool_calls_by_index: dict[int, int], max_bucket_to_log: int) -> None:
        self._tool_calls_by_index = dict(tool_calls_by_index)
        self._max_bucket_to_log = int(max_bucket_to_log)
        self._seen: set[int] = set()
        self._seen_unique_by_bucket: Counter[int] = Counter()
        self._total_by_bucket: Counter[int] = Counter(self._tool_calls_by_index.values())
        self._total = len(self._tool_calls_by_index)

    def _bucket_key(self, tool_calls: int) -> int:
        if tool_calls <= self._max_bucket_to_log:
            return tool_calls
        return self._max_bucket_to_log + 1

    def _record_sampled_indices(self, *, sampled_indices: list[int]) -> Counter[int]:
        sampled_by_bucket: Counter[int] = Counter()
        for idx in sampled_indices:
            tool_calls = self._tool_calls_by_index.get(idx)
            if tool_calls is None:
                continue
            bucket = self._bucket_key(tool_calls)
            sampled_by_bucket[bucket] += 1
            if idx in self._seen:
                continue
            self._seen.add(idx)
            self._seen_unique_by_bucket[bucket] += 1
        return sampled_by_bucket

    def advance(self, *, sampled_indices: list[int]) -> None:
        """Update internal coverage state without emitting metrics (used for resuming)."""
        self._record_sampled_indices(sampled_indices=sampled_indices)

    def update(self, *, step: int, sampled_indices: list[int]) -> dict[str, float]:
        sampled_by_bucket = self._record_sampled_indices(sampled_indices=sampled_indices)

        metrics: dict[str, float] = {
            "train/curriculum_seen_unique_total": float(len(self._seen)),
            "train/curriculum_seen_frac_total": float(len(self._seen) / max(1, self._total)),
        }

        for bucket in range(0, self._max_bucket_to_log + 2):
            if bucket == self._max_bucket_to_log + 1:
                suffix = f"{self._max_bucket_to_log + 1}_plus"
                denom = sum(v for k, v in self._total_by_bucket.items() if k > self._max_bucket_to_log)
            else:
                suffix = str(bucket)
                denom = self._total_by_bucket.get(bucket, 0)

            metrics[f"train/curriculum_batch_tool_calls_{suffix}"] = float(sampled_by_bucket.get(bucket, 0))
            metrics[f"train/curriculum_seen_unique_tool_calls_{suffix}"] = float(
                self._seen_unique_by_bucket.get(bucket, 0)
            )
            metrics[f"train/curriculum_seen_frac_tool_calls_{suffix}"] = float(
                self._seen_unique_by_bucket.get(bucket, 0) / max(1, denom)
            )

        if step % COVERAGE_PRINT_EVERY_STEPS == 0:
            hard_bucket = self._max_bucket_to_log + 1
            hard_total = sum(v for k, v in self._total_by_bucket.items() if k > self._max_bucket_to_log)
            print(
                "[coverage] "
                f"step={step} seen={len(self._seen)}/{self._total} "
                f"batch(tool_calls>={self._max_bucket_to_log + 1})={sampled_by_bucket.get(hard_bucket, 0)} "
                f"seen(tool_calls>={self._max_bucket_to_log + 1})={self._seen_unique_by_bucket.get(hard_bucket, 0)}/{hard_total}"
            )

        return metrics


def prefill_coverage_tracker(
    *,
    coverage: CurriculumCoverageTracker,
    initial_step: int,
    train_task_indices: list[int],
    tasks: list[Any],
    config: RunConfig,
    training_config: Any,
) -> None:
    """
    When resuming training mid-run, W&B logging uses the global step index, but the
    in-memory coverage tracker resets on process restart.

    To keep `train/curriculum_seen_*` continuous across restarts, replay the deterministic
    sampler for steps < `initial_step` and update coverage state without logging.
    """
    from art.utils import iterate_dataset

    if initial_step <= 0:
        return

    if config.curriculum is not None and config.curriculum.enabled:
        prefill_iter = iterate_curriculum(
            base_epoch_size=len(train_task_indices),
            groups_per_step=training_config.groups_per_step,
            num_epochs=training_config.num_epochs,
            initial_step=0,
            tasks=tasks,
            config=config,
            seed=config.seed,
            use_tqdm=False,
        )
    else:
        prefill_iter = iterate_dataset(
            train_task_indices,
            groups_per_step=training_config.groups_per_step,
            num_epochs=training_config.num_epochs,
            initial_step=0,
            use_tqdm=False,
        )

    for batch in prefill_iter:
        if batch.step >= initial_step:
            break
        coverage.advance(sampled_indices=list(batch.items))


def difficulty_for_step(*, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    return float(max(0.0, min(1.0, step / (total_steps - 1))))


def iterate_curriculum(
    *,
    base_epoch_size: int,
    groups_per_step: int,
    num_epochs: int,
    initial_step: int,
    tasks: list[Any],
    config: RunConfig,
    seed: int,
    use_tqdm: bool = True,
) -> Generator[DatasetBatch[int], None, None]:
    """
    Build deterministic curriculum batches.

    Keeps steps-per-epoch constant (based on `base_epoch_size`) by cycling through the eligible
    pool with coverage guarantees (no repeats until the pool is exhausted), and only repeating
    when the curriculum pool is smaller than the required number of draws.

    If `run.curriculum.sampling == "mixture"`, each step draws a fixed-size mixture across
    tool-call buckets (via `ToolCallsMixtureSampler`) instead of sampling uniformly from a
    single eligible pool.
    """
    from art.utils.iterate_dataset import DatasetBatch

    if base_epoch_size <= 0:
        return

    steps_per_epoch = math.ceil(base_epoch_size / groups_per_step)
    total_steps = steps_per_epoch * num_epochs

    curriculum = config.curriculum
    use_mixture = (
        curriculum is not None
        and curriculum.enabled
        and not config.task_ids
        and getattr(curriculum, "sampling", "unlock") == "mixture"
    )

    sampler = ShuffleBagSampler(seed=seed)
    mixture_sampler: ToolCallsMixtureSampler | None = None
    if use_mixture:
        assert curriculum is not None
        base_indices = get_task_indices(
            task_ids=config.task_ids,
            start_index=config.start_index,
            end_index=config.end_index,
            tasks=tasks,
            curriculum=None,
            difficulty=None,
            seed=seed,
        )
        mixture_sampler = ToolCallsMixtureSampler(
            tasks=tasks,
            indices=base_indices,
            curriculum=curriculum,
            seed=seed,
        )

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc="Iterating curriculum",
            unit="batch",
        )

    try:
        for global_step in range(total_steps):
            epoch = global_step // steps_per_epoch
            epoch_step = global_step % steps_per_epoch

            difficulty = difficulty_for_step(step=global_step, total_steps=total_steps)
            if mixture_sampler is not None:
                items = mixture_sampler.sample_batch(difficulty=difficulty, batch_size=groups_per_step)
            else:
                eligible = get_task_indices(
                    task_ids=config.task_ids,
                    start_index=config.start_index,
                    end_index=config.end_index,
                    tasks=tasks,
                    curriculum=config.curriculum,
                    difficulty=difficulty,
                    seed=seed,
                )
                items = sampler.sample_batch(eligible=eligible, batch_size=groups_per_step)
            if global_step < initial_step:
                continue
            yield DatasetBatch(step=global_step, epoch=epoch, epoch_step=epoch_step, items=items)
            if progress_bar:
                progress_bar.update(1)
    finally:
        if progress_bar:
            progress_bar.close()
