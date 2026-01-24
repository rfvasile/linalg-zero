import asyncio
import concurrent.futures
import copy
import logging
import os
import traceback
from datetime import datetime

import art
import wandb
from art.local import LocalBackend
from art.utils import iterate_dataset, limit_concurrency
from dotenv import load_dotenv

from linalg_zero.grpo.agents.tool_calling_agent import ToolCallingRLAgent
from linalg_zero.grpo.envs import get_env
from linalg_zero.grpo.general_rm import calculate_reward
from linalg_zero.grpo.rl_utils import (
    log_trajectory_to_openpipe,
    write_eval_trajectories,
)
from linalg_zero.grpo.run import agent_factory
from linalg_zero.grpo.task_selection import (
    get_task_indices,
)
from linalg_zero.grpo.types import LinAlgPolicyConfig, RunConfig, SolveResult
from linalg_zero.grpo.utils.checkpointing import archive_checkpoint, delete_checkpoints_keep_best
from linalg_zero.grpo.utils.curriculum import (
    COVERAGE_LOG_MAX_TOOL_CALLS_BUCKET,
    CurriculumCoverageTracker,
    iterate_curriculum,
    prefill_coverage_tracker,
)
from linalg_zero.grpo.utils.eval_metrics import (
    aggregate_retry_summaries,
    log_eval_aggregate,
    log_group_diversity,
    summarize_trajectories,
)
from linalg_zero.grpo.utils.hf_upload import push_experiment_dir_to_hf_sync
from linalg_zero.grpo.utils.trajectory_messages import clean_messages

# Load environment variables
load_dotenv(override=True)

# Suppress LiteLLM logging spam
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@limit_concurrency(256)
async def rollout_linalg_task(
    model: art.Model[LinAlgPolicyConfig],
    task_index: int,
    step: int = 0,
    phase: str = "train",
    reward_type: str = "real",
    is_shadow: bool = False,
) -> art.Trajectory:
    """
    Generate a trajectory for a single tau-bench task using the given model.
    This adapts the tau-bench evaluation loop for RL trajectory generation.
    Now truly async.
    """
    # print(f"Rolling out task {task_index} (step {step}, phase {phase})")
    config = copy.deepcopy(model.config.run_config)
    success_reward = 1.0
    if is_shadow:
        config.model = "gpt-4.1"
        config.model_provider = "openai"
        config.api_key = None
        config.base_url = None

    # Get isolated environment for this task
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        task_split=phase,
        dataset_path=config.dataset_path,
        task_index=task_index,
    )
    if config.add_no_think:
        env.wiki += "\n/no_think"

    # Create agent with the trainable model
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )

    if not isinstance(agent, ToolCallingRLAgent):
        raise TypeError("Agent must be a ToolCallingRLAgent")

    # Create trajectory object
    traj = art.Trajectory(
        messages_and_choices=[],
        tools=env.tools_info,
        reward=0,
        metadata={
            "task_index": str(task_index),
            "env": config.env,
            "training_step": str(step),
            "phase": phase,
            "model": model.name,
            "reward_type": config.reward_type,
            "is_shadow": str(is_shadow),
        },
    )

    try:
        # Run the agent on the task (now async call)
        result = await agent.solve(
            env=env,
            task_index=task_index,
            max_assistant_turns=config.max_assistant_turns,
        )
        # optimal_trajectory: 1 if the entire trajectory is optimal (correct answer + optimal efficiency)
        optimal_trajectory = 1 if abs(result.reward - success_reward) <= 1e-6 else 0

        # Convert result to trajectory format
        traj.reward, explanation = await calculate_reward(result, config)

        # Build metrics dictionary
        reward_info = result.info.get("reward_info") or {}
        info = reward_info.get("info") or {}
        outputs = info.get("outputs") or {}
        has_valid_outputs = "correctness_score" in outputs
        total_completion_tokens = result.info.get("total_completion_tokens")
        if not isinstance(total_completion_tokens, int | float):
            total_completion_tokens = result.info["avg_completion_tokens"] * result.info["total_steps"]
        traj.metrics = {
            "total_steps": result.info["total_steps"],
            "final_prompt_tokens": result.info["final_prompt_tokens"],
            "total_completion_tokens": float(total_completion_tokens),
            "avg_completion_tokens": result.info["avg_completion_tokens"],
            "max_completion_tokens": result.info["max_completion_tokens"],
            "forced_stop": result.info["forced_stop"],
            "optimal_trajectory": optimal_trajectory,
            "valid_trajectory": 1 if has_valid_outputs else 0,
        }

        if has_valid_outputs:
            traj.metrics.update({
                "correctness_score": outputs["correctness_score"],
                "format_score": outputs["format_score"],
                "tool_success_score": outputs["tool_success_score"],
                "efficiency_penalty": outputs["efficiency_penalty"],
                "num_turns": outputs["num_turns"],
                "expected_turns": outputs["expected_turns"],
                "turn_deviation": outputs["num_turns"] - outputs["expected_turns"],
            })
        traj.metadata.update(result.info)
        traj.metadata["reward"] = traj.reward
        traj.metadata["optimal_trajectory"] = traj.metrics["optimal_trajectory"]
        traj.metadata["judge_explanation"] = explanation

        if config.messages_only:
            traj.messages_and_choices = clean_messages(result.messages)
        else:
            traj.messages_and_choices = agent.create_messages_and_choices()
    except Exception as e:
        print(f"Error in rollout for task {task_index}: {e}")
        traj.reward = -1.0
        traj.metadata["error"] = str(e)
        traj.metadata["traceback"] = traceback.format_exc()
        traj.messages_and_choices = agent.create_messages_and_choices()
        result = SolveResult(
            reward=-1.0,
            info={"error": str(e)},
            messages=agent.messages,
            total_cost=0.0,
        )

    traj.finish()

    # Log to langfuse/openpipe
    try:
        await log_trajectory_to_openpipe(traj, result.messages)
    except Exception as e:
        print(f"Error logging trajectory to openpipe: {e}")

    # print(f"Finished rolling out task {task_index} (reward: {traj.reward})")
    return traj


async def evaluate_model(
    model: art.Model[LinAlgPolicyConfig],
    config: RunConfig,
    step: int,
    val_task_indices: list[int],
    split: str = "val",
) -> float:
    """Evaluate the model on a subset of tasks"""
    eval_retries = 1
    try:
        training_config = model.config.training_config
        if training_config is not None and getattr(training_config, "eval_retries", None) is not None:
            eval_retries = max(1, int(training_config.eval_retries))
    except Exception:
        eval_retries = 1

    print(f"Evaluating model on {len(val_task_indices)} tasks (passes={eval_retries})...")

    summaries: list[dict[str, float]] = []

    model_step = await model.get_step()
    eval_step = max(step, model_step)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(config.log_dir, exist_ok=True)

    for pass_idx in range(eval_retries):
        trajectories = await art.gather_trajectories(
            rollout_linalg_task(
                model=model,
                task_index=val_task_index,
                step=eval_step,
                phase=split,
                reward_type=config.reward_type,
            )
            for val_task_index in val_task_indices
        )
        output_path = os.path.join(
            config.log_dir,
            f"eval_trajectories_{split}_step_{eval_step}_pass_{pass_idx}_{run_id}.jsonl",
        )
        write_eval_trajectories(
            output_path=output_path,
            trajectories=trajectories,
            eval_step=eval_step,
            pass_idx=pass_idx,
            split=split,
        )
        print(f"Wrote eval trajectories to {output_path}")
        summaries.append(summarize_trajectories(trajectories))
        if eval_retries > 1:
            print(f"Eval pass {pass_idx + 1}/{eval_retries}: reward={summaries[-1].get('reward', float('nan')):.4f}")

    aggregate = aggregate_retry_summaries(summaries=summaries)
    log_eval_aggregate(split=split, step=eval_step, aggregate=aggregate)

    print(
        f"[{split}] step={eval_step} n={int(aggregate.get('n', 0))} "
        f"reward_mean={aggregate.get('reward_mean', float('nan')):.4f} "
        f"reward_retry_std={aggregate.get('reward_std', float('nan')):.4f}"
    )
    if "optimal_trajectory_mean" in aggregate:
        print(f"[{split}] optimal_trajectory_mean={aggregate.get('optimal_trajectory_mean', float('nan')):.4f}")

    # Return mean reward across retries (for callers that use the float).
    return float(aggregate.get("reward_mean", float("nan")))


async def test(model: art.TrainableModel[LinAlgPolicyConfig]):
    """Main evaluation loop"""
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    config = model.config.run_config
    training_config = model.config.training_config

    if training_config is None:
        raise ValueError("Training config is not set")

    register_kwargs = {}
    if model.config.training_config.chat_template is not None:
        register_kwargs["_openai_client_config"] = art.dev.OpenAIServerConfig(
            server_args=art.dev.ServerArgs(chat_template=model.config.training_config.chat_template)
        )

    with LocalBackend(in_process=config.in_process) as backend:
        # Resume/fork must happen *before* register() so the Unsloth/vLLM service
        # loads the correct LoRA adapter on startup.
        if model.config.run_config.resume:
            await backend._experimental_fork_checkpoint(
                model,
                from_model=model.config.run_config.resume_from,
                from_project=model.config.run_config.project,
                not_after_step=model.config.run_config.resume_step,
                verbose=True,
            )

        # Setup model with backend (starts the inference server + loads LoRA)
        await model.register(backend, **register_kwargs)

        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        config.base_model = model.base_model

        print("Loading training tasks...")

        # Load validation environment
        test_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="test",
            dataset_path=config.dataset_path,
        )

        test_task_indices = get_task_indices(
            task_ids=None,
            start_index=0,
            end_index=-1,
            tasks=test_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        print(f"Validation on {len(test_task_indices)} tasks")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, final_step, test_task_indices, split="test")
        print(f"Final average reward: {final_reward}")

        print("Evaluation complete!")


async def train(model: art.TrainableModel[LinAlgPolicyConfig]):  # noqa: C901
    """Main training loop adapted from art-e example"""
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    config = model.config.run_config
    training_config = model.config.training_config

    if training_config is None:
        raise ValueError("Training config is not set")

    register_kwargs = {}
    if model.config.training_config.chat_template is not None:
        register_kwargs["_openai_client_config"] = art.dev.OpenAIServerConfig(
            server_args=art.dev.ServerArgs(chat_template=model.config.training_config.chat_template)
        )

    with LocalBackend(in_process=config.in_process) as backend:
        # Resume/fork must happen *before* register() so the Unsloth/vLLM service
        # loads the correct LoRA adapter on startup.
        if model.config.run_config.resume:
            await backend._experimental_fork_checkpoint(
                model,
                from_model=model.config.run_config.resume_from,
                from_project=model.config.run_config.project,
                not_after_step=model.config.run_config.resume_step,
                verbose=True,
            )
        else:
            print("Will continue training from previous latest checkpoint")

        # Setup model with backend (starts the inference server + loads LoRA)
        await model.register(backend, **register_kwargs)

        config.api_key = model.inference_api_key
        config.base_url = model.inference_base_url
        config.base_model = model.base_model

        print("Loading training tasks...")
        # Get environment to access tasks
        train_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="train",
            dataset_path=config.dataset_path,
        )

        # Load validation environment
        val_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            task_split="val",
            dataset_path=config.dataset_path,
        )

        train_task_indices = get_task_indices(
            task_ids=config.task_ids,
            start_index=config.start_index,
            end_index=config.end_index,
            tasks=train_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        val_task_indices = get_task_indices(
            task_ids=config.val_task_ids,
            start_index=config.start_val_index,
            end_index=config.end_val_index,
            tasks=val_env.tasks,
            curriculum=None,
            difficulty=None,
            seed=config.seed,
        )

        print(f"Training on {len(train_task_indices)} tasks")
        print(f"Validation on {len(val_task_indices)} tasks")

        coverage: CurriculumCoverageTracker | None = None
        if config.curriculum is not None and config.curriculum.enabled:
            tool_calls_by_index = {idx: len(train_env.tasks[idx].actions) for idx in train_task_indices}
            coverage = CurriculumCoverageTracker(
                tool_calls_by_index=tool_calls_by_index,
                max_bucket_to_log=COVERAGE_LOG_MAX_TOOL_CALLS_BUCKET,
            )

        initial_step = await model.get_step()
        base_epoch_size = len(train_task_indices)
        if config.curriculum is not None and config.curriculum.enabled:
            if coverage is not None:
                prefill_coverage_tracker(
                    coverage=coverage,
                    initial_step=initial_step,
                    train_task_indices=train_task_indices,
                    tasks=train_env.tasks,
                    config=config,
                    training_config=training_config,
                )
            train_iterator = iterate_curriculum(
                base_epoch_size=base_epoch_size,
                groups_per_step=training_config.groups_per_step,
                num_epochs=training_config.num_epochs,
                initial_step=initial_step,
                tasks=train_env.tasks,
                config=config,
                seed=config.seed,
            )
        else:
            # Training iterator
            train_iterator = iterate_dataset(
                train_task_indices,
                groups_per_step=training_config.groups_per_step,
                num_epochs=training_config.num_epochs,
                initial_step=initial_step,
            )

        for batch in train_iterator:
            print(f"\n--- Training Step {batch.step} (Epoch {batch.epoch}, Step {batch.epoch_step}) ---")

            if coverage is not None:
                coverage_metrics = coverage.update(step=batch.step, sampled_indices=list(batch.items))
                if wandb.run is not None:
                    wandb.log(coverage_metrics, step=batch.step)

            # Evaluation
            if batch.step % training_config.eval_steps == 0 and not config.skip_eval:
                print(f"\n--- Evaluating at Step {batch.step} ---")
                await evaluate_model(model, config, batch.step, val_task_indices)
                archive_checkpoint(model=model, step=await model.get_step(), split="val")
                while True:
                    # proceed = input("Delete all previous checkpoints? (yes/no/exit): ").lower().strip()
                    proceed = "yes"

                    if proceed == "yes":
                        print("Deleting checkpoints...")
                        await delete_checkpoints_keep_best(model)
                        break
                    elif proceed == "no":
                        print("Skipping checkpoint deletion.")
                        break
                    elif proceed == "exit":
                        print("Exiting...")
                        if wandb.run is not None:
                            wandb.finish()
                        return
                    else:
                        print("Please type 'yes', 'no', or 'exit'.")

            # Generate trajectory groups
            print(f"Generating trajectories for {len(batch.items)} tasks...")
            groups = await art.gather_trajectory_groups(
                art.TrajectoryGroup(
                    rollout_linalg_task(
                        model=model,
                        task_index=task_index,
                        step=batch.step,
                        phase="train",
                        reward_type=config.reward_type,
                        is_shadow=config.add_shadow_trajectory
                        and rollout_idx % training_config.trajectories_per_group == 0,
                    )
                    for rollout_idx in range(training_config.trajectories_per_group)
                )
                for task_index in batch.items
            )
            # await model.log(groups, split="train")
            log_group_diversity(step=batch.step, groups=groups, split="train")

            # Training step
            print(f"Training on {len(groups)} trajectory groups...")
            dev_train_config: art.dev.TrainConfig = {
                "plot_tensors": config.plot_tensors,
                "importance_sampling_level": training_config.importance_sampling_level,
                "allow_training_without_logprobs": bool(config.messages_only),
                "scale_rewards": training_config.scale_rewards,
            }
            if training_config.epsilon is not None:
                dev_train_config["epsilon"] = training_config.epsilon
            if training_config.epsilon_high is not None:
                dev_train_config["epsilon_high"] = training_config.epsilon_high
            if training_config.max_negative_advantage_importance_sampling_weight is not None:
                dev_train_config["max_negative_advantage_importance_sampling_weight"] = (
                    training_config.max_negative_advantage_importance_sampling_weight
                )
            if training_config.truncated_importance_sampling is not None:
                dev_train_config["truncated_importance_sampling"] = training_config.truncated_importance_sampling
            await model.train(
                groups,
                config=art.TrainConfig(learning_rate=training_config.learning_rate, beta=training_config.beta),
                _config=dev_train_config,
            )
            if config.is_multi_gpu:
                await delete_checkpoints_keep_best(model)

            # Log progress
            total_reward = sum(sum(traj.reward for traj in group.trajectories) for group in groups)
            num_trajectories = sum(len(group.trajectories) for group in groups)
            avg_reward = total_reward / num_trajectories if num_trajectories > 0 else 0
            print(f"Step {batch.step}: Average training reward = {avg_reward}")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        final_step = await model.get_step()
        final_reward = await evaluate_model(model, config, final_step, val_task_indices)
        archive_checkpoint(model=model, step=await model.get_step(), split="val")
        wandb.finish()
        print(f"Final average reward: {final_reward}")

        # Optional post-training upload to HF Hub (enabled when `HF_HUB_NAMESPACE` is set).
        try:
            await asyncio.to_thread(push_experiment_dir_to_hf_sync, model=model)
        except Exception as e:
            print(f"[hf] Warning: post-training upload failed: {e}")

        print("Training completed!")
