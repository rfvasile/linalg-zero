from __future__ import annotations

import math
import random
from collections.abc import Sequence

from linalg_zero.grpo.types import CurriculumConfig, Task


class ShuffleBagSampler:
    """
    Coverage-guaranteeing sampler ("shuffle bag").

    Repeated calls to `sample_batch()` return indices from the eligible pool such that:
    - No index repeats until all currently-eligible indices have been returned once (a "cycle"),
      unless `batch_size` exceeds the pool size.
    - If the eligible pool grows over time (curriculum), newly added indices are injected into the
      current cycle so they are surfaced before any repeats.

    This is deterministic given the same `seed` and the same sequence of eligible pools.
    """

    def __init__(self, *, seed: int = 0, shuffle: bool = True) -> None:
        self._rng = random.Random(int(seed))
        self._shuffle = bool(shuffle)

        self._pool_set: set[int] = set()
        self._pool_order: list[int] = []

        self._seen_in_cycle: set[int] = set()
        self._remaining: list[int] = []
        self._remaining_set: set[int] = set()

    def _reset_cycle(self) -> None:
        self._seen_in_cycle.clear()
        self._remaining = list(self._pool_order)
        if self._shuffle and len(self._remaining) > 1:
            self._rng.shuffle(self._remaining)
        self._remaining_set = set(self._remaining)

    def _update_pool(self, *, eligible: Sequence[int]) -> None:
        eligible_list = list(eligible)
        eligible_set = set(eligible_list)
        if not eligible_set:
            raise ValueError("Cannot sample from an empty index set.")

        if not self._pool_set:
            self._pool_set = eligible_set
            self._pool_order = eligible_list
            self._reset_cycle()
            return

        added = [idx for idx in eligible_list if idx not in self._pool_set]
        removed = self._pool_set - eligible_set

        if not added and not removed:
            return

        if removed:
            self._pool_set.difference_update(removed)
            self._seen_in_cycle.difference_update(removed)
            if self._remaining:
                self._remaining = [idx for idx in self._remaining if idx not in removed]
                self._remaining_set.difference_update(removed)

        if added:
            self._pool_set.update(added)
            # Inject newly-eligible tasks into the current cycle so we don't defer them until the next cycle.
            for idx in added:
                if idx in self._seen_in_cycle or idx in self._remaining_set:
                    continue
                pos = self._rng.randrange(len(self._remaining) + 1)
                self._remaining.insert(pos, idx)
                self._remaining_set.add(idx)

        self._pool_order = eligible_list

        # If we removed indices, it's possible to end up with an empty remaining set mid-cycle.
        # Start a fresh cycle in that case.
        if not self._remaining:
            self._reset_cycle()

    def sample_batch(self, *, eligible: Sequence[int], batch_size: int) -> list[int]:
        """
        Sample a batch of indices with coverage guarantees.

        If `batch_size > len(eligible)`, repeats are unavoidable; this will still maximize coverage by
        cycling through the pool as many times as needed.
        """
        if batch_size <= 0:
            return []

        self._update_pool(eligible=eligible)

        out: list[int] = []
        while len(out) < batch_size:
            if not self._remaining:
                self._reset_cycle()
            idx = self._remaining.pop()
            self._remaining_set.remove(idx)
            self._seen_in_cycle.add(idx)
            out.append(idx)
        return out


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _deterministic_counts_from_probs(*, probs: Sequence[float], total: int) -> list[int]:
    if total <= 0:
        return [0 for _ in probs]

    cleaned: list[float] = [max(0.0, float(p)) for p in probs]
    mass = sum(cleaned)
    if mass <= 0.0:
        # Fall back to uniform allocation if the distribution degenerates.
        cleaned = [1.0 for _ in probs]
        mass = float(len(cleaned))

    normed = [p / mass for p in cleaned]
    expected = [p * total for p in normed]
    # Deterministic rounding: first take the floor of each expected count.
    counts = [math.floor(e) for e in expected]

    remainder = total - sum(counts)
    if remainder <= 0:
        return counts

    # Then allocate the leftover `remainder` to buckets with the largest fractional remainders
    # (`expected - floor(expected)`), not to buckets with the largest already-integer `counts`.
    fractional = [(expected[i] - counts[i], i) for i in range(len(counts))]
    fractional.sort(key=lambda item: (-item[0], item[1]))
    for _, idx in fractional[:remainder]:
        counts[idx] += 1
    return counts


class ToolCallsMixtureSampler:
    """
    Deterministic per-step mixture sampler over tool-call "difficulty" buckets.

    - Buckets are defined by `len(task.actions)` (teacher tool calls).
    - Each bucket uses its own ShuffleBagSampler to maximize coverage within the bucket.
    - Per-step mixture weights are computed from a Gaussian centered on a target tool-call
      count that increases linearly with `difficulty`.
    """

    def __init__(
        self,
        *,
        tasks: Sequence[Task],
        indices: Sequence[int],
        curriculum: CurriculumConfig,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if curriculum.metric != "tool_calls":
            raise ValueError(f"Unknown curriculum metric: {curriculum.metric!r}")

        self._tasks = tasks
        self._curriculum = curriculum
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._calls = 0

        base_indices = list(indices)
        if not base_indices:
            raise ValueError("Cannot create curriculum sampler from an empty index set.")

        buckets: dict[int, list[int]] = {}
        for idx in base_indices:
            tool_calls = len(tasks[idx].actions)
            buckets.setdefault(tool_calls, []).append(idx)

        self._min_tool_calls = min(buckets.keys())
        max_tool_calls_seen = max(buckets.keys())

        initial_tool_calls = max(self._min_tool_calls, int(curriculum.initial_max_tool_calls))
        requested_final = (
            max_tool_calls_seen
            if curriculum.final_max_tool_calls is None
            else min(int(curriculum.final_max_tool_calls), max_tool_calls_seen)
        )
        requested_final = max(self._min_tool_calls, requested_final)
        if requested_final < initial_tool_calls:
            raise ValueError(
                "Invalid curriculum: final_max_tool_calls must be >= initial_max_tool_calls "
                f"(got initial={initial_tool_calls}, final={requested_final})."
            )
        self._final_tool_calls = requested_final

        filtered_keys = sorted([k for k in buckets if k <= self._final_tool_calls])
        if not filtered_keys:
            raise ValueError("Curriculum sampler has no buckets after applying final_max_tool_calls filter.")

        self._bucket_keys = filtered_keys
        self._bucket_ordered: dict[int, list[int]] = {}
        self._bucket_samplers: dict[int, ShuffleBagSampler] = {}

        for tool_calls in self._bucket_keys:
            bucket = list(buckets[tool_calls])
            if self._shuffle and len(bucket) > 1:
                rng = random.Random(self._seed + (tool_calls + 1) * 1_000_003)
                rng.shuffle(bucket)
            self._bucket_ordered[tool_calls] = bucket
            self._bucket_samplers[tool_calls] = ShuffleBagSampler(
                seed=self._seed + (tool_calls + 1) * 2_000_033,
                shuffle=self._shuffle,
            )

    def _target_tool_calls(self, *, difficulty: float) -> float:
        difficulty = _clamp01(difficulty)
        start = float(max(self._min_tool_calls, int(self._curriculum.initial_max_tool_calls)))
        end = float(max(start, self._final_tool_calls))
        return float(max(self._min_tool_calls, min(end, start + difficulty * (end - start))))

    def _mixture_probs(self, *, difficulty: float) -> list[float]:
        target = self._target_tool_calls(difficulty=difficulty)
        sigma = float(getattr(self._curriculum, "mixture_sigma", 0.0))
        if sigma <= 0.0:
            # Hard assignment to the nearest bucket when sigma is disabled.
            nearest = min(self._bucket_keys, key=lambda tc: (abs(float(tc) - target), tc))
            return [1.0 if tc == nearest else 0.0 for tc in self._bucket_keys]

        denom = 2.0 * sigma * sigma
        if denom == 0.0:
            # Extremely small positive sigma can underflow `sigma * sigma` to 0.0.
            nearest = min(self._bucket_keys, key=lambda tc: (abs(float(tc) - target), tc))
            return [1.0 if tc == nearest else 0.0 for tc in self._bucket_keys]
        log_weights = [-((float(tc) - target) ** 2) / denom for tc in self._bucket_keys]
        max_log = max(log_weights)
        weights = [math.exp(w - max_log) for w in log_weights]
        mass = sum(weights)
        if mass <= 0.0:
            # Extremely small sigma can cause underflow; fall back to nearest-bucket.
            nearest = min(self._bucket_keys, key=lambda tc: (abs(float(tc) - target), tc))
            probs = [1.0 if tc == nearest else 0.0 for tc in self._bucket_keys]
        else:
            probs = [w / mass for w in weights]

        floor = float(getattr(self._curriculum, "mixture_min_prob_easiest", 0.0))
        floor = float(max(0.0, min(1.0, floor)))
        if floor <= 0.0:
            return probs

        p_easy = probs[0]
        if p_easy >= floor or p_easy >= 1.0:
            return probs

        remaining = 1.0 - p_easy
        if remaining <= 0.0:
            return [1.0] + [0.0 for _ in probs[1:]]

        scale = (1.0 - floor) / remaining
        adjusted = [floor] + [p * scale for p in probs[1:]]
        return adjusted

    def sample_batch(self, *, difficulty: float, batch_size: int) -> list[int]:
        if batch_size <= 0:
            return []

        difficulty = _clamp01(difficulty)

        probs = self._mixture_probs(difficulty=difficulty)
        # Convert continuous mixture probabilities into an exact integer allocation for this step.
        # Note: when a bucket's expected count is <1 (e.g., early 3-tool-call exposure with small `p3`),
        # deterministic rounding can "flicker" between 0 and 1 across steps; once the expected count
        # crosses 1 (as difficulty increases), that bucket will appear every step thereafter.
        counts = _deterministic_counts_from_probs(probs=probs, total=batch_size)

        frac = self._curriculum.fraction_at_start + difficulty * (
            self._curriculum.fraction_at_end - self._curriculum.fraction_at_start
        )
        frac = _clamp01(frac)

        out: list[int] = []
        for tool_calls, count in zip(self._bucket_keys, counts, strict=True):
            if count <= 0:
                continue

            ordered = self._bucket_ordered[tool_calls]
            k = math.ceil(frac * len(ordered))
            if k <= 0:
                k = 1
            eligible = ordered[:k]

            sampler = self._bucket_samplers[tool_calls]
            out.extend(sampler.sample_batch(eligible=eligible, batch_size=count))

        if len(out) != batch_size:
            raise RuntimeError(f"Mixture sampler produced {len(out)} indices, expected {batch_size}.")

        # Mix difficulties within the step so the batch isn't ordered by bucket.
        # Deterministic given seed and call order (which is driven by the training step).
        rng = random.Random(self._seed + 9_000_001 + self._calls)
        rng.shuffle(out)
        self._calls += 1
        return out


def get_task_indices(  # noqa: C901
    *,
    task_ids: list[int] | None,
    start_index: int,
    end_index: int,
    tasks_length: int | None = None,
    tasks: Sequence[Task] | None = None,
    curriculum: CurriculumConfig | None = None,
    difficulty: float | None = None,
    seed: int = 0,
) -> list[int]:
    """
    Return a list of task indices, optionally filtered by a curriculum.

    - If `task_ids` is provided, it always wins (curriculum is ignored).
    - Otherwise uses `[start_index, end_index)` (or full length if `end_index == -1`).
    - If `curriculum.enabled` and `difficulty` is provided, returns a deterministic subset
      that grows monotonically with `difficulty` (0.0 -> easiest subset, 1.0 -> full set).
    """
    if task_ids:
        return list(task_ids)

    if tasks_length is None:
        if tasks is None:
            raise ValueError("Must provide `tasks_length` or `tasks` when task_ids is not set.")
        tasks_length = len(tasks)

    actual_start = max(0, start_index)
    actual_end = tasks_length if end_index == -1 else min(end_index, tasks_length)
    base_indices = list(range(actual_start, max(actual_start, actual_end)))

    if curriculum is None or not curriculum.enabled or difficulty is None:
        return base_indices

    if tasks is None:
        raise ValueError("Curriculum selection requires `tasks` to be provided.")

    if curriculum.metric != "tool_calls":
        raise ValueError(f"Unknown curriculum metric: {curriculum.metric!r}")

    difficulty = float(max(0.0, min(1.0, difficulty)))

    tool_calls_by_index: dict[int, int] = {}
    max_tool_calls_seen = 0
    min_tool_calls_seen: int | None = None
    for idx in base_indices:
        tool_calls = len(tasks[idx].actions)
        tool_calls_by_index[idx] = tool_calls
        max_tool_calls_seen = max(max_tool_calls_seen, tool_calls)
        min_tool_calls_seen = tool_calls if min_tool_calls_seen is None else min(min_tool_calls_seen, tool_calls)

    initial_max = max(0, int(curriculum.initial_max_tool_calls))
    final_max = (
        max_tool_calls_seen if curriculum.final_max_tool_calls is None else int(curriculum.final_max_tool_calls)
    )
    if final_max < initial_max:
        final_max = initial_max

    levels = max(1, final_max - initial_max + 1)
    level = min(levels - 1, math.floor(difficulty * levels))
    allowed_max = initial_max + level
    if min_tool_calls_seen is not None:
        allowed_max = max(allowed_max, min_tool_calls_seen)

    # Fraction of tasks to expose within each included difficulty bucket.
    frac = curriculum.fraction_at_start + difficulty * (curriculum.fraction_at_end - curriculum.fraction_at_start)
    frac = float(max(0.0, min(1.0, frac)))

    buckets: dict[int, list[int]] = {}
    for idx in base_indices:
        tool_calls = tool_calls_by_index[idx]
        if tool_calls <= allowed_max:
            buckets.setdefault(tool_calls, []).append(idx)

    ordered_buckets: dict[int, list[int]] = {}
    for tool_calls in sorted(buckets.keys()):
        bucket = list(buckets[tool_calls])
        rng = random.Random(seed + (tool_calls + 1) * 1_000_003)
        rng.shuffle(bucket)
        ordered_buckets[tool_calls] = bucket

    # Stable "easiest-first" ordering so that when new buckets are unlocked,
    # previously selected tasks remain selected.
    easiest_first: list[int] = []
    for tool_calls in sorted(ordered_buckets.keys()):
        easiest_first.extend(ordered_buckets[tool_calls])

    # Baseline floor (picked from easiest-first) to keep selection non-empty and monotonic.
    min_total = max(0, int(curriculum.min_total_tasks))
    baseline = set(easiest_first[: min(min_total, len(easiest_first))])

    # Fraction-based selection (per bucket), chosen as prefixes of deterministic per-bucket order.
    fraction_selected: set[int] = set()
    for tool_calls in sorted(ordered_buckets.keys()):
        bucket = ordered_buckets[tool_calls]
        k = math.ceil(frac * len(bucket))
        if k <= 0:
            continue
        fraction_selected.update(bucket[:k])

    selected_set = baseline | fraction_selected
    if not selected_set and easiest_first:
        selected_set = {easiest_first[0]}

    return [idx for idx in easiest_first if idx in selected_set]
