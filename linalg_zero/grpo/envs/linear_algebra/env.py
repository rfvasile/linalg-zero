import random
from functools import cache

from linalg_zero.grpo.envs.base import Env
from linalg_zero.grpo.envs.linear_algebra.tasks import load_tasks
from linalg_zero.grpo.envs.linear_algebra.tools import ALL_TOOLS
from linalg_zero.grpo.envs.user import UserStrategy
from linalg_zero.grpo.types import RESPOND_ACTION_NAME, RewardOutputInfo, RewardResult, Task
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.system_prompts import get_sft_system_prompt

from .compute_reward import answer_correct, think_correct, validate_answer


@cache
def _load_tasks_cached(dataset_path: str, split: str) -> tuple[Task, ...]:
    """Cache tasks loading to avoid repeated HuggingFace calls."""
    return tuple(load_tasks(dataset_path, split=split))


class LinearAlgebraEnv(Env):
    """Linear algebra environment for mathematical reasoning tasks."""

    def __init__(
        self,
        dataset_path: str,
        user_strategy: str | UserStrategy = UserStrategy.MATH,
        task_split: str = "test",
        task_index: int | None = None,
    ):
        # Load tasks based on split
        tasks = self._get_tasks(dataset_path, task_split)

        super().__init__(
            data_load_func=lambda: {},  # Linear algebra doesn't need external data
            tools=ALL_TOOLS,
            tasks=list(tasks),
            wiki=get_sft_system_prompt(),
            rules=[],
            user_strategy=user_strategy,
            task_index=task_index,
            parser=XMLParser(),
            task_split=task_split,
        )
        self.terminate_tools = []

    def _get_tasks(self, hf_path: str, task_split: str) -> tuple[Task, ...]:
        """Get tasks for the specified split."""
        split_mapping = {
            "test": "test",
            "train": "train",
            "val": "validation",
        }

        if task_split not in split_mapping:
            raise ValueError(f"Unknown task split: {task_split}. Valid splits: {list(split_mapping.keys())}")

        split = split_mapping[task_split]
        return _load_tasks_cached(hf_path, split)

    def reasoning_depth_reward(self) -> float:
        """Reward appropriate reasoning depth."""
        contents = [a.content for a in self.actions if isinstance(a.content, str)]
        if not contents:
            return 0.0

        # Approximate response length to a range of approx. 100 and 550 tokens.
        rewards = [1.0 if 650 < len(c) < 3500 else 0.5 for c in contents]
        return sum(rewards) / len(rewards)

    def tool_success_reward(self) -> float:
        """Track successful tool execution, not just presence."""
        if not self.actions:
            return 0.0

        tool_attempts = 0
        successful_executions = 0

        # Check each tool call's cached observation from Env.step instead of re-invoking tools.
        for action in self.actions[:-1]:
            obs = action._observation
            if not isinstance(obs, str):
                continue

            tool_attempts += 1
            # Treat both explicit tool errors and unknown actions as failures.
            if not (obs.startswith("Error:") or obs.startswith("Unknown action")):
                successful_executions += 1

        return successful_executions / tool_attempts if tool_attempts > 0 else 0.0

    async def calculate_reward(self) -> RewardResult:
        """Composite reward that combines several components:

        - Correctness (primary signal)
        - Formatting of thoughts/answers
        - Tool-call success (to avoid "dummy" tool calls)
        - Efficiency (penalizes over/under-using tools)
        """
        # Handle degenerate / structurally invalid trajectories similarly to calculate_reward_old.
        if not self.actions:
            return RewardResult(
                reward=-1.0,
                info=RewardOutputInfo(
                    r_outputs=-1.0,
                    outputs={"structural_error": "no_actions", "answer_found": False},
                ),
                actions=[],
            )

        tool_calls = self.actions[:-1]
        answer = self.actions[-1]

        # No tool calls: treat as maximally lazy.
        if len(tool_calls) == 0:
            return RewardResult(
                reward=-0.3,
                info=RewardOutputInfo(
                    r_outputs=-0.3,
                    outputs={"structural_error": "no_tool_calls", "answer_found": False},
                ),
                actions=self.actions,
            )

        # Final step must be a respond action.
        if answer.name != RESPOND_ACTION_NAME:
            return RewardResult(
                reward=-1.0,
                info=RewardOutputInfo(
                    r_outputs=-1.0,
                    outputs={"structural_error": "no_respond_action", "answer_found": False},
                ),
                actions=self.actions,
            )

        # Compute individual components once so they are consistent between
        # the scalar reward and the logged info.
        correctness = self.correctness_reward()
        format_score = self.format_reward()
        tool_success = self.tool_success_reward()
        efficiency_penalty = self.efficiency_penalty()

        # Weights for each component.
        correctness_weight = 1.0
        format_weight = 0.1
        tool_success_weight = 0.1
        efficiency_weight = 0.1

        total_reward = (
            correctness_weight * correctness
            + format_weight * format_score
            + tool_success_weight * tool_success
            - efficiency_weight * efficiency_penalty
        ) / 1.2

        return RewardResult(
            reward=total_reward,
            info=RewardOutputInfo(
                r_outputs=total_reward,
                outputs={
                    "answer_found": bool(correctness),
                    "correctness_score": correctness,
                    "format_score": format_score,
                    "tool_success_score": tool_success,
                    "efficiency_penalty": efficiency_penalty,
                    "num_turns": len(tool_calls),
                    "expected_turns": len(self.task.actions),
                },
            ),
            actions=tool_calls,
        )

    def get_jitter(self) -> float:
        return random.uniform(0, 1e-4)

    def correctness_reward(self) -> float:
        completion = self.actions[-1].content
        if completion is None:
            return 0.0
        return 1.0 if validate_answer(ground_truth=self.task.outputs[0], completion=completion) else 0.0

    def efficiency_penalty(self) -> float:
        """Return an absolute penalty based on deviation from the expected number of tool calls.

        0.0  -> used exactly the expected number of tool calls
        3.0  -> maximum deviation (3+ extra tool calls beyond expected)
        """
        expected_turns = len(self.task.actions)
        actual_turns = len(self.actions[:-1])

        if expected_turns == 0:
            return 0.0

        diff = max(0, actual_turns - expected_turns)
        return min(float(diff), 3)

    def format_reward(self) -> float:
        """
        Reward proper formatting.

        Intermediate turns (tool calls):
        - Check for <think> tag in content
        - Weight: 1.0 per turn

        Final turn (answer):
        - Check for <think> tag in content
        - Check for <answer> tag in content
        - Weight: 3.0 total (answer tag is more important than think)

        Returns:
            Float between 0.0 and 1.0 (weighted average of correct formats)
        """
        if not self.actions:
            return 0.0

        correct_formats = 0.0
        total_weight = 0.0

        for action in self.actions[:-1]:
            has_think = think_correct(completion=action.content or "")
            if has_think:
                correct_formats += 1.0
            total_weight += 1.0

        # Check final turn (weight = 3.0; answer tag is twice as important as think)
        final_action = self.actions[-1]
        final_completion = final_action.content or ""
        has_think = think_correct(completion=final_completion)
        has_answer = answer_correct(completion=final_completion)

        if has_think:
            correct_formats += 1.0
        if has_answer:
            correct_formats += 2.0
        total_weight += 3.0

        return correct_formats / total_weight if total_weight > 0 else 0.0
