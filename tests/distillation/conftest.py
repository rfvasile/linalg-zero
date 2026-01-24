from unittest.mock import MagicMock

import pytest
from distilabel.models.llms.base import LLM

from linalg_zero.distillation.components.multi_turn_generation_base import MultiTurnWithToolUseBase
from linalg_zero.shared.lib import get_lib_fn_names


@pytest.fixture
def base() -> MultiTurnWithToolUseBase:
    """A minimal MultiTurnWithToolUseBase configured for testing pure methods.

    The pure methods we exercise (_execute, extract_*, check_final_answers) don't
    invoke self.llm, so a MagicMock satisfies the field type without doing real I/O.
    """
    return MultiTurnWithToolUseBase(
        llm=MagicMock(spec=LLM),
        n_turns=3,
        library=get_lib_fn_names(),
        model_type="default",
        structured_output=False,
    )
