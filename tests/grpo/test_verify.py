import pytest

from linalg_zero.grpo.verify import parse_string, verify_answers


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("42", 42),
        ("-17.5", -17.5),
        ("[[1, 2], [3, 4]]", [[1, 2], [3, 4]]),
        ("[1, 2, 3]", [1, 2, 3]),
        ("", None),  # empty
        ("The answer is 42", None),  # not a literal — no flexible extraction
        ("42.5.6", None),  # malformed number
        ("[[1, 2], [3, 4", None),  # unbalanced brackets
    ],
)
def test_parse_string(raw: str, expected: object) -> None:
    assert parse_string(raw) == expected, f"parse_string({raw!r})"


@pytest.mark.parametrize(
    "ground_truth, answer, expected",
    [
        ("42", "42", True),  # exact
        ("42", "43", False),  # mismatch
        ("[[1, 2], [3, 4]]", "[[1, 2], [3, 4]]", True),
        ("[[1, 2], [3, 4]]", "[[4, 3], [2, 1]]", False),
        ("2", "2.0", True),  # int/float equivalence
        ("[[1, 2], [3, 4]]", "[[1.0, 2.0], [3.0, 4.0]]", True),
        ("0", "-0", True),
    ],
)
def test_verify_answers_matching(ground_truth: str, answer: str, expected: bool) -> None:
    assert verify_answers(parse_string(ground_truth), parse_string(answer)) is expected


@pytest.mark.parametrize("gt, ans", [(None, 42), (42, None), (None, None)])
def test_verify_answers_none_is_false(gt: object, ans: object) -> None:
    assert verify_answers(gt, ans) is False  # type: ignore[arg-type]
