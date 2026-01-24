import abc
import enum
from typing import ClassVar


class BaseUserSimulationEnv(abc.ABC):
    metadata: ClassVar[dict[str, object]] = {}

    @abc.abstractmethod
    async def reset(self, instruction: str | None = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def step(self, content: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class LocalSimulationEnv(BaseUserSimulationEnv):
    async def reset(self, instruction: str | None = None) -> str:
        return f"{instruction}\n"

    async def step(self, content: str) -> str:
        return f"{content}\n"

    def get_total_cost(self) -> float:
        return 0


class UserStrategy(enum.Enum):
    MATH = "mathematician"


def load_user(
    user_strategy: str | UserStrategy,
) -> BaseUserSimulationEnv:
    if isinstance(user_strategy, str):
        user_strategy = UserStrategy(user_strategy)
    if user_strategy == UserStrategy.MATH:
        return LocalSimulationEnv()
    raise ValueError(f"Unknown user strategy {user_strategy}")
