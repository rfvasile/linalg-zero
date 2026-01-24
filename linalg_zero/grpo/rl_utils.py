import contextlib
import dataclasses
import json
import os
import time
import uuid
from typing import Any

import art
from art.trajectories import MetadataValue
from openpipe.client import AsyncOpenPipe


def json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def write_eval_trajectories(
    *,
    output_path: str,
    trajectories: list[art.Trajectory],
    eval_step: int,
    pass_idx: int,
    split: str,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for trajectory_idx, traj in enumerate(trajectories):
            task_index: int | str | None = None
            if isinstance(traj.metadata, dict):
                task_index = traj.metadata.get("task_index")
            if isinstance(task_index, str):
                with contextlib.suppress(ValueError):
                    task_index = int(task_index)
            if task_index is None:
                task_index = trajectory_idx
            try:
                reward_value: float | str = float(traj.reward)
            except (TypeError, ValueError):
                reward_value = traj.reward

            record = {
                "split": split,
                "step": eval_step,
                "pass": pass_idx,
                "trajectory_idx": trajectory_idx,
                "task_index": task_index,
                "reward": reward_value,
                "metrics": traj.metrics if isinstance(traj.metrics, dict) else {},
                "metadata": traj.metadata if isinstance(traj.metadata, dict) else {},
                "messages_and_choices": traj.messages_and_choices or [],
            }
            f.write(json.dumps(record, default=json_default, ensure_ascii=True) + "\n")


def string_to_string_dict(metadata: dict[str, Any]) -> dict[str, str]:
    string_dict = {}
    for key, value in metadata.items():
        if isinstance(value, MetadataValue):
            string_dict[key] = str(value)
    return string_dict


def create_response_payload(response_str: str | None = None) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-dummy-{uuid.uuid4()!s}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": response_str or "Dummy Response",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def log_trajectory_to_openpipe(
    traj: art.Trajectory,
    messages: list[dict[str, Any]],
    response_str: str | None = None,
) -> None:
    """Push one trajectory to OpenPipe with task_idx and step for comparison."""
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])
    report_payload_metrics = string_to_string_dict(traj.metadata)
    resp_payload = create_response_payload(response_str=response_str)
    traj.metadata["completion_id"] = resp_payload["id"]

    await op_client.report(
        req_payload={
            "model": traj.metadata["model"],
            "messages": messages,
            "tools": traj.tools,
            "metadata": report_payload_metrics,
        },
        resp_payload=resp_payload,
        status_code=200,
    )
    await op_client.base_client._client_wrapper.httpx_client.aclose()
