import random
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from linalg_zero.grpo.envs.tool import Tool
from linalg_zero.grpo.envs.user import UserStrategy, load_user
from linalg_zero.grpo.types import (RESPOND_ACTION_NAME, Action, EnvInfo,
                                    EnvResetResponse, EnvResponse,
                                    RewardActionInfo, RewardOutputInfo,
                                    RewardResult, Task)
from linalg_zero.grpo.verifiers.xml_parser import XMLParser

ToHashable = str | int | float | dict[str, "ToHashable"] | list["ToHashable"] | set["ToHashable"]
HashableItem = str | int | float | tuple[Any, ...]


def to_hashable(item: ToHashable) -> HashableItem:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(
    value: HashableItem,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


class Env:
    def __init__(
        self,
        data_load_func: Callable[[], dict[str, Any]],
        tools: list[type[Tool]],
        tasks: list[Task],
        wiki: str,
        rules: list[str],
        user_strategy: str | UserStrategy,
        task_index: int | None = None,
        parser: XMLParser | None = None,
        task_split: str = "train",
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.parser = parser
        self.data = data_load_func()
        self.tools_map: dict[str, type[Tool]] = {tool.get_info()["function"]["name"]: tool for tool in tools}
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools: list[str] = []
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1)
        self.task = tasks[self.task_index]
        self.wiki = wiki
        self.rules = rules
        self.user = load_user(user_strategy=user_strategy)
        self.actions: list[Action] = []
        self.task_split = task_split

    async def reset(self, task_index: int | None = None) -> EnvResetResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks) - 1)
        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []
        initial_observation = await self.user.reset(instruction=self.task.instruction)
        return EnvResetResponse(observation=initial_observation, info=EnvInfo(task=self.task, source="user"))

    async def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)

        info = EnvInfo(task=self.task)
        reward: float = 0.0
        done = False
        if action.name == RESPOND_ACTION_NAME:
            content = action.content
            if content is None:
                raise ValueError("Respond action must include content")
            observation = await self.user.step(content)
            info.source = "user"
            done = True
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:
                observation = f"Error: {e}"
            info.source = action.name
            if action.name in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        action._observation = observation
        if done:
            reward_res = await self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()
        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def get_data_hash(self) -> str:
        return consistent_hash(to_hashable(self.data))

    async def calculate_reward(self) -> RewardResult:
        data_hash = self.get_data_hash()
        reward = 1.0
        actions = [action for action in self.task.actions if action.name != RESPOND_ACTION_NAME]

        # Check if the database changes are correct. If they are not correct, then we set the reward to 0.
        # TODO: cache gt_data_hash in tasks.py (low priority)
        self.data = self.data_load_func()
        for action in self.task.actions:
            if action.name not in self.terminate_tools:
                await self.step(action)
        gt_data_hash = self.get_data_hash()
        actions_ok = data_hash == gt_data_hash
        reward_info: RewardActionInfo | RewardOutputInfo = RewardActionInfo(
            r_actions=actions_ok, gt_data_hash=gt_data_hash
        )
        if not actions_ok:
            reward = 0.0

        if len(self.task.outputs) > 0:
            # check outputs
            r_outputs = 1.0
            outputs: dict[str, bool | str | float | int] = {}
            for output in self.task.outputs:
                output_str = str(output).lower().replace(",", "")
                found = False
                for action in self.actions:
                    content = action.kwargs.get("content")
                    if (
                        action.name == RESPOND_ACTION_NAME
                        and isinstance(content, str)
                        and output_str in content.lower().replace(",", "")
                    ):
                        found = True
                        break
                outputs[str(output)] = found
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            reward_info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)

        return RewardResult(reward=reward, info=reward_info, actions=actions)
