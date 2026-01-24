import abc

from linalg_zero.grpo.envs.base import Env
from linalg_zero.grpo.types import SolveResult


class Agent(abc.ABC):
    @abc.abstractmethod
    async def solve(self, env: Env, task_index: int | None = None, max_assistant_turns: int = 30) -> SolveResult:
        raise NotImplementedError
