from linalg_zero.grpo.envs.base import Env
from linalg_zero.grpo.envs.linear_algebra.env import LinearAlgebraEnv
from linalg_zero.grpo.envs.user import UserStrategy


def get_env(
    env_name: str,
    user_strategy: str | UserStrategy,
    task_split: str,
    dataset_path: str,
    task_index: int | None = None,
) -> Env:
    if env_name == "linear_algebra":
        return LinearAlgebraEnv(
            dataset_path=dataset_path,
            user_strategy=user_strategy,
            task_split=task_split,
            task_index=task_index,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
