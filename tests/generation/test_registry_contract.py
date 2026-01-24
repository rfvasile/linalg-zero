import pytest

from linalg_zero.generator.models import DifficultyCategory, Topic
from linalg_zero.generator.registry import FactoryRegistry

ALL_DIFFICULTIES = list(DifficultyCategory)


def test_every_registered_task_resolves_to_a_callable(default_registry: FactoryRegistry) -> None:
    for topic in default_registry.list_topics():
        for task in default_registry.list_problem_types(topic):
            factory = default_registry.get_factory(topic, task)
            assert callable(factory), f"Factory for {topic}/{task} is not callable"


@pytest.mark.parametrize("difficulty", ALL_DIFFICULTIES)
def test_every_difficulty_has_at_least_one_factory(
    default_registry: FactoryRegistry, difficulty: DifficultyCategory
) -> None:
    # An empty bucket would make generate_exact_per_factory silently produce zero questions.
    factories = default_registry.get_factories_by_difficulty(Topic.LINEAR_ALGEBRA, difficulty)
    assert len(factories) > 0, f"No factories registered at difficulty {difficulty.name}"


def test_composite_components_are_well_formed(default_registry: FactoryRegistry) -> None:
    """First component must be independent, subsequent ones dependent — this is what
    dependent generators rely on to know whether to generate inputs or receive them."""
    composite_tasks = [
        t for t in default_registry.list_problem_types(Topic.LINEAR_ALGEBRA) if t.name.startswith(("TWO_", "THREE_"))
    ]
    assert composite_tasks, "Expected composite tasks in the default registry"

    for task in composite_tasks:
        components = default_registry.get_composite_components(Topic.LINEAR_ALGEBRA, task)
        assert len(components) >= 2, f"{task.name}: composite must have ≥2 components, got {len(components)}"

        first_task, first_independent = components[0]
        assert first_independent, f"{task.name}: first component {first_task.name} must be independent"

        for step_task, is_independent in components[1:]:
            assert not is_independent, f"{task.name}: subsequent component {step_task.name} must be dependent"
