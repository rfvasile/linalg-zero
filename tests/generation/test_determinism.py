import random

import numpy as np

import linalg_zero.generator.difficulty_config as difficulty_config
from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.generator.registry import FactoryRegistry, create_default_registry
from linalg_zero.generator.utils import set_seed


def _run(seed: int, registry: FactoryRegistry) -> list[tuple[str, str]]:
    set_seed(seed)
    difficulty_config.DETERMINISTIC_BASE_SEED = seed
    generator = DatasetGenerator(topic=Topic.LINEAR_ALGEBRA, registry=registry)
    dataset = generator.generate_exact_for_categories({
        DifficultyCategory.ONE_TOOL_CALL: 1,
        DifficultyCategory.TWO_TOOL_CALLS: 1,
    })
    return [(q.question, q.answer) for q in dataset]


def test_same_seed_yields_identical_dataset() -> None:
    # Fresh registry per run so registry-level state would expose itself.
    first = _run(seed=1234, registry=create_default_registry())
    second = _run(seed=1234, registry=create_default_registry())
    assert first == second, "Two runs at the same seed produced different datasets"


def test_different_seeds_produce_different_datasets() -> None:
    # Guards against a recurrence of the `from … import DETERMINISTIC_BASE_SEED`
    # binding bug, which made --seed silently no-op.
    first = _run(seed=1234, registry=create_default_registry())
    second = _run(seed=5678, registry=create_default_registry())
    assert first != second, "Different seeds produced identical datasets — seed is not propagating"


def test_factory_output_is_invariant_to_intervening_rng_noise() -> None:
    """Cross-phase invariance — the property the per-question reseed scheme exists for.

    Question N must be identical whether or not other RNG-consuming work runs between
    factory calls, so the analysis and generation scripts produce aligned datasets.
    """

    def _three_clean() -> list[tuple[str, str]]:
        registry = create_default_registry()
        factory = registry.get_factory(Topic.LINEAR_ALGEBRA, Task.ONE_DETERMINANT)
        return [(q.question, q.answer) for q in (factory() for _ in range(3))]

    def _three_with_noise() -> list[tuple[str, str]]:
        registry = create_default_registry()
        factory = registry.get_factory(Topic.LINEAR_ALGEBRA, Task.ONE_DETERMINANT)
        results = []
        for _ in range(3):
            random.random()
            random.randint(0, 1000)
            np.random.rand(7)
            results.append(factory())
        return [(q.question, q.answer) for q in results]

    clean = _three_clean()
    noisy = _three_with_noise()
    assert clean == noisy, "Per-question reseed failed to isolate factory output from intervening RNG state"
