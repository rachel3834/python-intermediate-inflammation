"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected)

@pytest.mark.parametrize(
    "test, expected",
    [
        (np.ones((3,2),dtype='int'), [1, 1]),
        ([[1, 1], [1, 10], [1, 1]], [1, 10]),
    ])
def test_daily_max_value(test, expected):
    """
    Test that the maximum function identifies the highest entry
    in an array.
    """
    from inflammation.models import daily_max

    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        (np.ones((3,2),dtype='int'), [1, 1]),
        ([[1, 1], [1, -10], [1, 1]], [1, -10]),
    ])
def test_daily_min_value(test, expected):
    """
    Test that the minimum function correctly identifies the lowest
    entry in an array.
    """
    from inflammation.models import daily_min

    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])
