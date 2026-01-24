from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

import wandb

from linalg_zero.grpo.utils.trajectory_messages import extract_tool_name_sequence

if TYPE_CHECKING:
    import art


def log_group_diversity(*, step: int, groups: list[art.TrajectoryGroup], split: str) -> None:
    if not groups:
        return

    group_reward_stds: list[float] = []
    group_first_tool_diversity: list[float] = []
    first_tool_names: list[str] = []

    for group in groups:
        rewards = [float(traj.reward) for traj in group.trajectories]
        group_reward_stds.append(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0)

        first_tools: list[str] = []
        for traj in group.trajectories:
            seq = extract_tool_name_sequence(traj)
            if seq:
                first_tools.append(seq[0])
        if first_tools:
            first_tool_names.extend(first_tools)
            group_first_tool_diversity.append(len(set(first_tools)) / len(first_tools))
        else:
            group_first_tool_diversity.append(0.0)

    top_first_tools = Counter(first_tool_names).most_common(5)
    reward_std_p95 = (
        float(statistics.quantiles(group_reward_stds, n=20)[-1])
        if len(group_reward_stds) >= 20
        else float(max(group_reward_stds))
    )
    metrics: dict[str, float | str] = {
        f"{split}/group_reward_std_mean": float(statistics.mean(group_reward_stds)),
        f"{split}/group_reward_std_p95": reward_std_p95,
        f"{split}/group_first_tool_diversity_mean": float(statistics.mean(group_first_tool_diversity)),
        f"{split}/group_first_tool_diversity_min": float(min(group_first_tool_diversity)),
    }
    for idx, (name, count) in enumerate(top_first_tools, start=1):
        metrics[f"{split}/first_tool_top{idx}_name"] = name
        metrics[f"{split}/first_tool_top{idx}_count"] = float(count)

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    if step % 10 == 0:
        print(
            f"[{split}] step={step} reward_std_mean={metrics[f'{split}/group_reward_std_mean']:.4f} "
            f"first_tool_div_mean={metrics[f'{split}/group_first_tool_diversity_mean']:.3f} "
            f"top_first_tools={top_first_tools}"
        )


def log_eval_aggregate(*, split: str, step: int, aggregate: dict[str, float]) -> None:
    """
    Log a single aggregated evaluation point (mean across `eval_retries`) to W&B.

    This avoids writing multiple `val/*` points at the same training step.
    """
    if wandb.run is None:
        return

    payload: dict[str, float] = {}
    for k, v in aggregate.items():
        if not isinstance(v, int | float):
            continue
        if k == "n":
            continue
        if k.endswith("_mean"):
            base = k[: -len("_mean")]
            payload[f"{split}/{base}"] = float(v)

    wandb.log(payload, step=step)


def summarize_trajectories(trajectories: list[art.Trajectory]) -> dict[str, float]:  # noqa: C901
    """Compute simple mean metrics from a list of trajectories (used for eval retries)."""
    if not trajectories:
        return {}

    rewards = [float(t.reward) for t in trajectories]
    summary: dict[str, float] = {
        "reward": float(statistics.mean(rewards)),
        "reward_std_dev": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
    }

    keys: set[str] = set()
    for t in trajectories:
        if t.metrics:
            keys.update(t.metrics.keys())

    for key in sorted(keys):
        vals: list[float] = []
        for t in trajectories:
            if not t.metrics:
                continue
            v = t.metrics.get(key)
            if isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, int | float):
                vals.append(float(v))
        if vals:
            summary[key] = float(statistics.mean(vals))

    errors = 0
    for t in trajectories:
        if isinstance((t.metadata or {}).get("error"), str):
            errors += 1
    summary["exception_rate"] = errors / len(trajectories)
    return summary


def aggregate_retry_summaries(*, summaries: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate multiple eval summaries (each summary is already averaged over tasks).

    Produces mean/min/max/std across retries for each key plus `n` retries.
    """
    if not summaries:
        return {"n": 0.0}

    keys: set[str] = set()
    for s in summaries:
        keys.update(s.keys())

    out: dict[str, float] = {"n": float(len(summaries))}
    for key in sorted(keys):
        vals = [s[key] for s in summaries if isinstance(s.get(key), int | float)]
        if not vals:
            continue
        out[f"{key}_mean"] = float(statistics.mean(vals))
        out[f"{key}_min"] = float(min(vals))
        out[f"{key}_max"] = float(max(vals))
        out[f"{key}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    return out
