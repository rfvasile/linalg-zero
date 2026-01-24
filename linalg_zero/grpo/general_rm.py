from linalg_zero.grpo.types import RunConfig, SolveResult


async def calculate_reward(result: SolveResult, config: RunConfig) -> tuple[float, str]:
    # Forced stops (e.g. token-budget exhaustion) always score -1, regardless of reward type.
    if result.info["forced_stop"]:
        return -1, "Max token trajectory"

    if config.reward_type == "real":
        return result.reward, "real_reward"

    raise ValueError(f"Invalid reward type: {config.reward_type}")
