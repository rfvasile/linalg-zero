import zlib
from collections.abc import Callable
from typing import Any

from linalg_zero.generator import difficulty_config
from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.template_engine import TemplateEngine
from linalg_zero.generator.utils import set_seed


def _question_seed(base: int, problem_type: Task, topic: Topic, i: int) -> int:
    """Stable per-question seed; drives cross-phase invariance."""
    key = f"{base}|{problem_type.value}|{topic.value}|{i}".encode()
    return zlib.crc32(key)


def create_composite_factory(
    components: list[ProblemComponent],
    composition_strategy: CompositionStrategy,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
) -> Callable[[], Question]:
    """
    Factory function for creating composite problem generators.
    """

    counter = {"i": 0}

    def factory() -> Question:
        if difficulty_config.DETERMINISTIC_MODE:
            set_seed(
                _question_seed(
                    difficulty_config.DETERMINISTIC_BASE_SEED,
                    problem_type,
                    topic,
                    counter["i"],
                )
            )
        generator = CompositeProblem(
            components=components,
            composition_strategy=composition_strategy,
            template_engine=TemplateEngine(),
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )
        try:
            return generator.generate()
        finally:
            counter["i"] += 1

    return factory


def create_sympy_factory(
    generator_class: type,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
    entropy: EntropyConstraints,
    gen_constraints: GenerationConstraints | None = None,
    **kwargs: Any,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """
    counter = {"i": 0}

    def factory() -> Question:
        if difficulty_config.DETERMINISTIC_MODE:
            set_seed(
                _question_seed(
                    difficulty_config.DETERMINISTIC_BASE_SEED,
                    problem_type,
                    topic,
                    counter["i"],
                )
            )
        value = entropy.sample_entropy()
        generator: SympyProblemGenerator = generator_class(
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            template_engine=TemplateEngine(),
            entropy=value,
            local_index=0,
            gen_constraints=gen_constraints,
            constraints={},
            **kwargs,
        )
        try:
            return generator.generate()
        finally:
            counter["i"] += 1

    return factory
