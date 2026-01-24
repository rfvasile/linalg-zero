import json

import pytest

from linalg_zero.distillation.components.models import DIAG_PREFIX
from linalg_zero.grpo.verifiers.xml_parser import XMLParser

TOOLS = ["determinant"]


@pytest.fixture
def parser() -> XMLParser:
    return XMLParser()


def _tool_call(body: str) -> str:
    return f"<think>t</think><tool_call>{body}</tool_call>"


def _valid_tool_json() -> str:
    return json.dumps({"name": "determinant", "arguments": {"matrix": [[1, 2], [3, 4]]}})


def test_extract_tag_contents_all_vs_last(parser: XMLParser) -> None:
    msg = "<think>a</think><think>b</think>"
    assert parser.extract_tag_contents(msg, "think") == ["a", "b"]
    assert parser.extract_tag_contents(msg, "think", last_only=True) == ["b"]
    assert parser.extract_tag_contents("<answer>x", "answer") == []  # unclosed → nothing
    assert parser.extract_tag_contents("", "think") == []


@pytest.mark.parametrize(
    "message, expected_valid",
    [
        (_tool_call(_valid_tool_json()), True),  # think + tool
        ("<think>t</think><answer>-2.0</answer>", True),  # think + answer
        ("<tool_call>{}</tool_call>", False),  # missing think
        ("<think>t</think><tool_call>{}</tool_call><answer>x</answer>", False),  # both blocks
        ("plain text", False),  # neither
        ("junk<think>t</think><answer>1</answer>", True),  # only the last <think> onward must match
    ],
)
def test_format_validation(parser: XMLParser, message: str, expected_valid: bool) -> None:
    analysis = parser.analyze_message(message, tool_names=TOOLS)
    assert analysis["is_valid_think_then_tool_or_answer"] is expected_valid


@pytest.mark.parametrize(
    "body, json_valid, name, name_known",
    [
        (_valid_tool_json(), True, "determinant", True),
        ("{not valid json}", False, "", None),
        (
            json.dumps({"name": "determinant", "arguments": json.dumps({"matrix": [[1, 2]]})}),
            True,
            "determinant",
            True,
        ),
        (json.dumps({"name": "nope", "arguments": {}}), True, "nope", False),  # unknown tool
    ],
)
def test_tool_json_parsing(parser: XMLParser, body: str, json_valid: bool, name: str, name_known: bool | None) -> None:
    tool = parser.analyze_message(_tool_call(body), tool_names=TOOLS)["tool"]
    assert tool["json_valid"] is json_valid
    assert tool["name"] == name
    assert tool["name_known"] is name_known


def test_answer_policy(parser: XMLParser) -> None:
    answer = "<think>t</think><answer>-2.0</answer>"
    tool_ctx = [{"role": "tool", "content": "-2.0"}]
    # answer is valid only when the preceding message is a tool response
    assert parser.analyze_message_in_context(tool_ctx, answer, tool_names=TOOLS)["answer_policy_valid"] is True
    assert parser.analyze_message_in_context([], answer, tool_names=TOOLS)["answer_policy_valid"] is False
    # an injected diagnostic user message between the tool response and the answer is skipped
    diag_ctx = [{"role": "tool", "content": "r"}, {"role": "user", "content": f"{DIAG_PREFIX} hint"}]
    assert parser.analyze_message_in_context(diag_ctx, answer, tool_names=TOOLS)["answer_policy_valid"] is True
    # a tool-call turn carries no answer, so the answer policy never gates it
    assert (
        parser.analyze_message_in_context([], _tool_call(_valid_tool_json()), tool_names=TOOLS)["has_answer"] is False
    )


@pytest.mark.parametrize(
    "message, context, expected_reason",
    [
        ("<think>a</think><think>b</think><answer>1</answer>", [], "multiple <think> blocks (nested/repeated)"),
        ("<think>t</think><tool_call>{}</tool_call><answer>1</answer>", [], "both tool call and answer present"),
        ("plain text", [], "no <think>/<tool_call>/<answer> blocks"),
        (f"<tool_call>{_valid_tool_json()}</tool_call>", [], "missing <think>"),
        (_tool_call("{not json}"), [], "invalid tool JSON"),
        (_tool_call(json.dumps({"name": "nope", "arguments": {}})), [], "unknown tool name"),
        ("<think>t</think><answer>1</answer>", [], "answer without tool response"),
    ],
)
def test_failure_reason(parser: XMLParser, message: str, context: list[dict], expected_reason: str) -> None:
    analysis = parser.analyze_message_in_context(context, message, tool_names=TOOLS)
    assert parser.get_analysis_failure_reason(analysis, TOOLS) == expected_reason
