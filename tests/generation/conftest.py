import pytest

from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.models import DifficultyCategory, Question, Topic
from linalg_zero.generator.registry import FactoryRegistry, create_default_registry


@pytest.fixture(scope="module")
def default_registry() -> FactoryRegistry:
    return create_default_registry()


@pytest.fixture
def small_dataset(default_registry: FactoryRegistry) -> list[Question]:
    """One question per registered factory at every difficulty."""
    generator = DatasetGenerator(topic=Topic.LINEAR_ALGEBRA, registry=default_registry)
    return generator.generate_exact_for_categories({
        DifficultyCategory.ONE_TOOL_CALL: 1,
        DifficultyCategory.TWO_TOOL_CALLS: 1,
        DifficultyCategory.THREE_TOOL_CALLS: 1,
    })
