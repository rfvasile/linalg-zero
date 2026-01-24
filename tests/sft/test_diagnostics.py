import json

from linalg_zero.sft.diagnostics import DiagnosticTracker
from linalg_zero.sft.tool_evaluation import EvaluationState


def _state(
    *, messages: list[dict], sample: dict | None, strict: float = 0.0, partial: float = 0.0, answer: str | None = None
) -> EvaluationState:
    state = EvaluationState()
    state.messages = messages
    state.sample = sample
    state.strict_format_match = strict
    state.partial_format_score = partial
    state.generated_answer = answer
    return state


def test_loss_metrics_match_hand_computed_counts() -> None:
    # One sample expecting 2 tool calls; the model emits 1 tool call + 1 (wrong) answer.
    conversation = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "compute the determinant"},
        {
            "role": "assistant",
            "content": '<think>step</think><tool_call>{"name":"determinant","arguments":{"matrix":[[1,2],[3,4]]}}</tool_call>',
        },
        {"role": "tool", "content": "-2.0"},
        {"role": "assistant", "content": "<think>done</think><answer>999</answer>"},
    ]
    sample = {"stepwise_ground_truths": json.dumps(["s1", "s2"]), "ground_truth": "-2.0"}

    tracker = DiagnosticTracker()
    tracker.update(_state(messages=conversation, sample=sample))
    metrics = tracker.calculate_loss_metrics()

    assert metrics["format_accuracy"] == (1 + 1) / (2 + 1)  # (actual_tool+actual_answer)/(expected_tool+samples)
    assert metrics["format_tool_call_accuracy"] == 1 / 2  # one of two expected tool calls
    assert metrics["format_answer_accuracy"] == 1 / 1  # one answer attempt for one sample
    assert metrics["answer_accuracy"] == 0.0  # 999 ≠ -2.0


def test_loss_metrics_empty_tracker_is_all_zero() -> None:
    metrics = DiagnosticTracker().calculate_loss_metrics()
    assert set(metrics.values()) == {0.0}


def test_progress_info_empty_tracker() -> None:
    assert DiagnosticTracker().get_progress_info() == {"strict": "0.000", "partial": "0.000", "correct": "0.000"}


def test_progress_info_averages_and_excludes_none_answers() -> None:
    tracker = DiagnosticTracker()
    tracker.update(_state(messages=[], sample={}, strict=1.0, partial=3.0, answer="x"))
    tracker.update(_state(messages=[], sample={}, strict=0.0, partial=1.0, answer=None))

    assert tracker.get_progress_info() == {"strict": "0.500", "partial": "2.000", "correct": "0.500"}
