import pytest

from linalg_zero.shared.lib import determinant


def test_determinant_two_by_two() -> None:
    result = determinant([[1, 2], [3, 4]])
    assert result == -2
    assert isinstance(result, float)


def test_determinant_diagonal() -> None:
    assert determinant([[2, 0], [0, 3]]) == 6


def test_determinant_identity() -> None:
    assert determinant([[1, 0], [0, 1]]) == 1


def test_determinant_one_by_one() -> None:
    assert determinant([[5]]) == 5


def test_determinant_non_square_raises() -> None:
    with pytest.raises(ValueError, match="Matrix must be square"):
        determinant([[1, 2, 3], [4, 5, 6]])
