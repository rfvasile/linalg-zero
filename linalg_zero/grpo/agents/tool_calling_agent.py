import json
from typing import Any

from art.utils import limit_concurrency
from art.utils.litellm import convert_litellm_choice_to_openai
from linalg_zero.grpo.agents.base import Agent
from linalg_zero.grpo.envs.base import Env
from linalg_zero.grpo.types import RESPOND_ACTION_NAME, Action, SolveResult
from litellm import Choices, acompletion
from litellm.types.utils import ModelResponse
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    reraise=True,
)
@limit_concurrency(n=128)
async def acompletion_with_limit_concurrency(*args: Any, **kwargs: Any) -> ModelResponse:
    return await acompletion(*args, **kwargs)


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.messages: list[dict[str, Any]] = []

    async def llm_completion(self, messages: list[dict[str, Any]], deterministic: bool = False) -> ModelResponse:
        temperature = 0.0 if deterministic else self.temperature
        completion_obj = await acompletion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            tools=self.tools_info,
            temperature=temperature,
        )
        assert isinstance(completion_obj, ModelResponse), "Completion object is not a ModelResponse"
        return completion_obj

    async def solve(self, env: Env, task_index: int | None = None, max_assistant_turns: int = 30) -> SolveResult:
        total_cost = 0.0
        env_reset_res = await env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        self.messages = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        final_prompt_tokens = 0
        total_completion_tokens = 0
        max_completion_tokens = 0
        forced_stop = True
        curr_step_number = 0
        for curr_step_number in range(max_assistant_turns):  # noqa: B007
            res = await self.llm_completion(self.messages, deterministic=env.task_split != "train")
            final_prompt_tokens = res.usage.prompt_tokens
            total_completion_tokens += res.usage.completion_tokens
            max_completion_tokens = max(max_completion_tokens, res.usage.completion_tokens)
            next_message = res.choices[0].message.model_dump()
            if (
                "tool_calls" in next_message
                and next_message["tool_calls"] is not None
                and len(next_message["tool_calls"]) > 0
                and next_message["tool_calls"][0]["function"] is not None
            ):
                next_message["tool_calls"] = next_message["tool_calls"][:1]
            self.messages.append(next_message)

            total_cost += res._hidden_params.get("response_cost") or 0.0
            action = message_to_action(next_message, res)
            env_response = await env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "name": next_message["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation,
                })

            if env_response.done:
                forced_stop = False
                break
            if final_prompt_tokens > 20000 or res.choices[0].finish_reason == "length":
                break
        info["total_steps"] = curr_step_number + 1
        info["total_completion_tokens"] = int(total_completion_tokens)
        info["avg_completion_tokens"] = total_completion_tokens / info["total_steps"]
        info["max_completion_tokens"] = max_completion_tokens
        info["final_prompt_tokens"] = final_prompt_tokens
        info["forced_stop"] = 1 if forced_stop else 0
        return SolveResult(
            reward=reward,
            info=info,
            messages=self.messages,
            total_cost=total_cost,
        )


class ToolCallingRLAgent(ToolCallingAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.base_model = kwargs.get("base_model")
        self.max_completion_tokens = kwargs.get("max_completion_tokens")
        self.skip_special_tokens = kwargs.get("skip_special_tokens")
        self.top_p = kwargs.get("top_p")
        self.repetition_penalty = kwargs.get("repetition_penalty")
        self.stop = kwargs.get("stop")
        self.seed = kwargs.get("seed")
        self.choices: list[Any] = []

    async def llm_completion(self, messages: list[dict[str, Any]], deterministic: bool = False) -> ModelResponse:
        temperature = 0.0 if deterministic else self.temperature
        do_sample = not deterministic
        top_p = None if deterministic else self.top_p
        # For training we want *stochastic* sampling across rollouts.
        # Passing a fixed seed can make rollouts nearly deterministic for identical prompts,
        # collapsing within-group diversity (bad for GRPO advantage estimation).
        seed = self.seed if deterministic else None
        request_kwargs: dict[str, Any] = {}
        if seed is not None:
            request_kwargs["seed"] = seed
        response = await acompletion_with_limit_concurrency(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            tools=self.tools_info,
            temperature=temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=self.repetition_penalty,
            logprobs=self.provider != "openai",
            extra_body={"skip_special_tokens": self.skip_special_tokens},
            stop=self.stop,
            **request_kwargs,
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            # if "Qwen3-" in self.base_model
            # else {},
        )
        assert isinstance(response, ModelResponse), f"Response is not a ModelResponse: {response}"
        choice = response.choices[0]
        assert isinstance(choice, Choices), f"Choice is not a Choices object: {choice}"
        self.choices.append(convert_litellm_choice_to_openai(choice))
        return response

    def create_messages_and_choices(self) -> list[Any]:
        messages_and_choices: list[Any] = []
        choice_idx = 0
        is_qwen3 = bool(self.base_model and "Qwen3-" in self.base_model)
        for message in self.messages:
            if message["role"] == "assistant":
                choice = self.choices[choice_idx]
                if hasattr(choice.message, "content") and choice.message.content is None and is_qwen3:
                    choice.message.content = ""
                messages_and_choices.append(choice)
                choice_idx += 1
            else:
                if is_qwen3:
                    if "content" in message and message["content"] is None:
                        message["content"] = ""
                    for key in list(message):
                        if message[key] is None:
                            message.pop(key)
                messages_and_choices.append(message)
        return messages_and_choices


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    """
    Robustly parse tool call arguments into a dict.

    Handles:
      - None -> {}
      - dict -> as-is
      - JSON string -> dict
      - double-encoded JSON string -> dict
    Falls back to {} on any failure.
    """
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {}

    current: Any = raw_args
    for _ in range(2):
        try:
            parsed = json.loads(current)
        except Exception:
            break
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str) and parsed != current:
            current = parsed
            continue
        break

    return {}


def message_to_action(message: dict[str, Any], res: Any) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        raw_args = tool_call["function"].get("arguments")
        kwargs = _parse_tool_arguments(raw_args)
        return Action(
            name=tool_call["function"]["name"],
            kwargs=kwargs,
            content=message["content"],
            completion_tokens=res.usage.completion_tokens,
        )
    return Action(
        name=RESPOND_ACTION_NAME, content=message["content"], kwargs={}, completion_tokens=res.usage.completion_tokens
    )
