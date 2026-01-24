from collections.abc import Iterator

import pytest

import linalg_zero.generator.difficulty_config as difficulty_config
from linalg_zero.generator.utils import set_seed

DEFAULT_TEST_SEED = 42


@pytest.fixture(autouse=True)
def seeded_rng() -> Iterator[None]:
    """Reset Python/NumPy/SymPy RNGs and the per-question reseed base before each test."""
    original_base = difficulty_config.DETERMINISTIC_BASE_SEED
    set_seed(DEFAULT_TEST_SEED)
    difficulty_config.DETERMINISTIC_BASE_SEED = DEFAULT_TEST_SEED
    try:
        yield
    finally:
        difficulty_config.DETERMINISTIC_BASE_SEED = original_base
