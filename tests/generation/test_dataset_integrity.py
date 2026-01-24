from linalg_zero.generator.models import Question
from linalg_zero.generator.utils import verify_dataset


def test_full_pipeline_produces_internally_consistent_dataset(
    small_dataset: list[Question],
) -> None:
    assert small_dataset, "small_dataset fixture produced no questions"

    results = verify_dataset(small_dataset)

    assert results["verified_questions"] == len(small_dataset)
    assert results["golden_verifications"] == len(small_dataset)
    assert results["stepwise_verifications"] >= len(small_dataset)
