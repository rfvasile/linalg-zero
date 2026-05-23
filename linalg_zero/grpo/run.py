import asyncio
import json
import multiprocessing
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from typing import Any

from langfuse import Langfuse
from litellm import provider_list

from linalg_zero.grpo.agents.base import Agent
from linalg_zero.grpo.envs import get_env
from linalg_zero.grpo.envs.user import UserStrategy
from linalg_zero.grpo.types import EnvRunResult, RunConfig

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:.*",
    category=UserWarning,
)


def run(config: RunConfig) -> list[EnvRunResult]:
    assert config.env in ["retail", "airline", "linear_algebra"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    success_reward = 1.0
    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_strategy}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        task_split=config.task_split,
        dataset_path=config.dataset_path,
    )
    end_index = len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    results: list[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")
    for i in range(config.num_trials):
        trial = i
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        # --- 2. helper -------
        def log_trace_to_langfuse(env_result: EnvRunResult, task_idx: int, cfg: RunConfig) -> None:
            """
            Push one full conversation to Langfuse.
            """
            # 2-a create / update the trace
            trace = langfuse.trace(
                name=f"{cfg.env}-task-{env_result.task_id}-{env_result.trial}",
                input=env_result.info,
                output=env_result.traj,
            )
            # 2-c attach numeric reward
            trace.score(name="reward", value=env_result.reward)

        def _run(idx: int, _trial: int = trial) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                task_split=config.task_split,
                dataset_path=config.dataset_path,
                task_index=idx,
            )
            agent = agent_factory(
                tools_info=isolated_env.tools_info,
                wiki=isolated_env.wiki,
                config=config,
            )

            print(f"Running task {idx}")
            try:
                res = asyncio.run(
                    agent.solve(
                        env=isolated_env,
                        task_index=idx,
                    )
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=_trial,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=_trial,
                )
            log_trace_to_langfuse(result, idx, config)
            print(
                "✅" if abs(result.reward - success_reward) <= 1e-6 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path) as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump([*data, result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results, success_reward=success_reward)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    langfuse.flush()
    return results


def agent_factory(tools_info: list[dict[str, Any]], wiki: str, config: RunConfig) -> Agent:
    if config.agent_strategy == "tool-calling-rl":
        from linalg_zero.grpo.agents.tool_calling_agent import ToolCallingRLAgent

        return ToolCallingRLAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            api_key=config.api_key,
            base_url=config.base_url,
            base_model=config.base_model,
            max_completion_tokens=config.max_completion_tokens,
            skip_special_tokens=config.skip_special_tokens,
            stop=config.stop,
        )
    elif config.agent_strategy == "tool-calling":
        from linalg_zero.grpo.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: list[EnvRunResult], success_reward: float = 1.0) -> None:
    def is_successful(reward: float) -> bool:
        return abs(reward - success_reward) <= 1e-6

    num_trials = len({r.trial for r in results})
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0.0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
