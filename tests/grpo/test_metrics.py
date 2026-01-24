from dataclasses import dataclass, field
from typing import Any

from linalg_zero.grpo.utils.curriculum import CurriculumCoverageTracker, difficulty_for_step
from linalg_zero.grpo.utils.eval_metrics import aggregate_retry_summaries, summarize_trajectories


@dataclass
class _Traj:
    """Minimal stand-in for art.Trajectory exposing the fields summarize_trajectories reads."""

    reward: float
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def test_summarize_trajectories_aggregates_and_coerces() -> None:
    trajs = [
        _Traj(1.0, {"correct": True, "steps": 2}),
        _Traj(0.0, {"correct": False, "steps": 4}, {"error": "boom"}),
    ]
    summary = summarize_trajectories(trajs)
    assert summary["reward"] == 0.5
    assert summary["reward_std_dev"] == 0.5
    assert summary["correct"] == 0.5  # bool metrics coerced to 0.0/1.0 then averaged
    assert summary["steps"] == 3.0
    assert summary["exception_rate"] == 0.5  # one trajectory carries a string error


def test_summarize_trajectories_edge_cases() -> None:
    assert summarize_trajectories([]) == {}
    single = summarize_trajectories([_Traj(0.7)])
    assert single["reward"] == 0.7
    assert single["reward_std_dev"] == 0.0  # pstdev undefined for n=1 → 0.0


def test_aggregate_retry_summaries() -> None:
    out = aggregate_retry_summaries(summaries=[{"reward": 1.0}, {"reward": 0.0, "x": 5.0}])
    assert out["n"] == 2.0
    assert out["reward_min"] <= out["reward_mean"] <= out["reward_max"]
    assert out["reward_std"] >= 0.0
    # a key present in only one summary is aggregated over the values that exist
    assert out["x_mean"] == 5.0 and out["x_std"] == 0.0


def test_aggregate_retry_summaries_empty() -> None:
    assert aggregate_retry_summaries(summaries=[]) == {"n": 0.0}


def test_curriculum_coverage_update_fractions_bounded() -> None:
    tracker = CurriculumCoverageTracker(tool_calls_by_index={0: 1, 1: 1, 2: 2, 3: 3}, max_bucket_to_log=2)
    metrics = tracker.update(step=1, sampled_indices=[0, 2, 3])
    assert metrics["train/curriculum_seen_frac_total"] == 0.75  # 3 of 4 indices seen
    # the over-threshold bucket is folded into a single "_plus" key, fraction still bounded
    assert metrics["train/curriculum_seen_frac_tool_calls_3_plus"] <= 1.0


def test_difficulty_for_step() -> None:
    assert difficulty_for_step(step=0, total_steps=10) == 0.0
    assert difficulty_for_step(step=9, total_steps=10) == 1.0
    assert difficulty_for_step(step=0, total_steps=1) == 1.0  # degenerate horizon
    assert difficulty_for_step(step=3, total_steps=10) < difficulty_for_step(step=7, total_steps=10)
