import ast
import json
from dataclasses import dataclass
from typing import Any

from linalg_zero.shared.types import LibTypes
from pydantic import BaseModel


def _get_config_store() -> Any | None:
    try:
        from hydra.core.config_store import ConfigStore
    except ModuleNotFoundError:  # pragma: no cover
        return None
    return ConfigStore


ConfigStore = _get_config_store()

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: dict[str, Any]
    content: str | None = None
    _observation: str | None = None
    completion_tokens: int = 0


class Task(BaseModel):
    user_id: str
    actions: list[Action]
    instruction: str
    outputs: list[LibTypes]

    @classmethod
    def from_dataset_entry(cls, entry: dict[str, Any]) -> "Task":
        stepwise = json.loads(entry["stepwise_ground_truths"])
        return cls(
            user_id=entry.get("user_id", "user"),
            actions=[
                Action(name=fn_name, kwargs={"matrix": ground_truth})
                for step in stepwise
                for fn_name, ground_truth in step.items()
            ],
            instruction=entry["query"],
            outputs=[ast.literal_eval(entry["ground_truth"])],
        )


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: dict[str, bool | str | float | int]


class RewardActionInfo(BaseModel):
    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    reward: float
    info: RewardOutputInfo | RewardActionInfo
    actions: list[Action]


class SolveResult(BaseModel):
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    total_cost: float | None = None


class EnvInfo(BaseModel):
    task: Task
    source: str | None = None
    user_cost: float | None = None
    reward_info: RewardResult | None = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    task_id: int
    reward: float
    info: dict[str, Any]
    traj: list[dict[str, Any]]
    trial: int


@dataclass
class CurriculumConfig:
    """
    Task curriculum configuration.

    Currently supports a simple "tool_calls" curriculum where difficulty is approximated by
    the number of teacher tool calls in the task (`len(task.actions)`).

    Sampling strategies:
    - "unlock": deterministic subset that grows with difficulty (legacy behavior)
    - "mixture": per-step fixed-size mixture across tool-call buckets (reduces end-of-epoch unlock plateaus)
    """

    enabled: bool = False
    metric: str = "tool_calls"
    initial_max_tool_calls: int = 1
    final_max_tool_calls: int | None = None
    fraction_at_start: float = 0.25
    fraction_at_end: float = 1.0
    min_total_tasks: int = 1
    sampling: str = "mixture"
    mixture_sigma: float = 0.5
    mixture_min_prob_easiest: float = 0.1


@dataclass
class RunConfig:
    project_id: str
    project: str
    model_provider: str
    dataset_path: str
    model: str = "gpt-4.1"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    temperature: float = 0.0
    top_p: float | None = 1.0
    repetition_penalty: float = 1.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: list[int] | None = None
    start_val_index: int = 0
    end_val_index: int = -1
    val_task_ids: list[int] | None = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: str | None = None
    # art related configs
    api_key: str | None = None
    base_url: str | None = None
    reward_type: str = "real"
    judge_model: str = "o3"
    max_assistant_turns: int = 30
    skip_eval: bool = False
    add_shadow_trajectory: bool = False
    messages_only: bool = False
    base_model: str = "unsloth/Qwen2.5-14B-Instruct"
    is_multi_gpu: bool = False
    add_no_think: bool = False
    plot_tensors: bool = False
    in_process: bool = False
    max_completion_tokens: int = 1024
    skip_special_tokens: bool = False
    stop: list[str] | None = None
    resume: bool = False
    resume_step: int | None = None
    resume_from: str | None = None
    curriculum: CurriculumConfig | None = None


@dataclass
class LinAlgTrainingConfig:
    """Training configuration for ART RL on tau-bench tasks"""

    chat_template: str | None = None
    trajectories_per_group: int = 6
    groups_per_step: int = 10
    learning_rate: float = 1.2e-5
    beta: float = 0.0
    eval_steps: int = 10
    val_set_size: int = 85
    eval_retries: int = 5
    num_epochs: int = 50
    train_mode: str = "sync_rl"
    importance_sampling_level: str = "token"  # or "sequence"
    scale_rewards: bool | None = True
    epsilon: float | None = None
    epsilon_high: float | None = None
    max_negative_advantage_importance_sampling_weight: float | None = None
    truncated_importance_sampling: float | None = None


class LinAlgPolicyConfig(BaseModel):
    """Policy configuration for tau-bench agent"""

    # Run config
    run_config: RunConfig

    # Training configuration
    training_config: LinAlgTrainingConfig


# Note: Both EngineArgs and TorchtuneArgs from art.dev are TypedDicts, not dataclasses,
# so we cannot register them with ConfigStore. Use plain dict configuration in YAML instead.
if ConfigStore is not None:  # pragma: no cover
    cs = ConfigStore.instance()
    cs.store(name="training_schema", node=LinAlgTrainingConfig, group="training")
    cs.store(name="run_schema", node=RunConfig, group="run")
