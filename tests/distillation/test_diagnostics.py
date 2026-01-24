import pytest

from linalg_zero.distillation.components.diagnostics import Diagnostics
from linalg_zero.distillation.components.models import DIAG_PREFIX, ModelType


@pytest.fixture
def diag() -> Diagnostics:
    return Diagnostics(model_type=ModelType.DEFAULT)


def _diag_user(content_suffix: str = "old") -> dict:
    return {"role": "user", "content": f"{DIAG_PREFIX} {content_suffix}"}


def test_apply_hint_default_cap_replaces_existing_diagnostic(diag: Diagnostics) -> None:
    # max_hints=1 → existing diagnostic dropped, new one appended.
    conversation = [
        {"role": "user", "content": "original query"},
        {"role": "assistant", "content": "malformed"},
        _diag_user("old hint"),
    ]
    diag.apply_hint(conversation, hint=f"{DIAG_PREFIX} new hint", max_hints=1)

    diagnostics = [m for m in conversation if diag.is_diagnostic_user_message(m)]
    assert len(diagnostics) == 1
    assert diagnostics[0]["content"].endswith("new hint")


def test_apply_hint_max_hints_2_retains_last_existing(diag: Diagnostics) -> None:
    # Exercises the `existing[-keep_count:]` retention path (the [-0:] guard).
    conversation = [
        {"role": "user", "content": "original query"},
        {"role": "assistant", "content": "malformed_1"},
        _diag_user("hint_1"),
        {"role": "assistant", "content": "malformed_2"},
        _diag_user("hint_2"),
    ]
    diag.apply_hint(conversation, hint=f"{DIAG_PREFIX} hint_3", max_hints=2)

    diagnostic_contents = [m["content"] for m in conversation if diag.is_diagnostic_user_message(m)]
    assert len(diagnostic_contents) == 2
    assert diagnostic_contents[0].endswith("hint_2")
    assert diagnostic_contents[1].endswith("hint_3")


def test_remove_hint_messages_drops_diagnostics_but_keeps_original_query(diag: Diagnostics) -> None:
    # Default model_type has append_policy()=False, so preceding assistants are NOT removed.
    conversation = [
        {"role": "user", "content": "original query"},
        {"role": "assistant", "content": "malformed"},
        _diag_user("hint"),
        {"role": "assistant", "content": "ok"},
    ]
    cleaned = diag.remove_hint_messages(conversation)
    assert cleaned == [
        {"role": "user", "content": "original query"},
        {"role": "assistant", "content": "malformed"},
        {"role": "assistant", "content": "ok"},
    ]
