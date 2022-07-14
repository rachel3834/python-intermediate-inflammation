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


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                None
        ),
        (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                None
        ),
        (
                [[1, 1, 1], [-1, -1, -1], [-1, -1, -1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ValueError
        ),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
                None
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            'I am a test string',
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[4.5, 2.1, 3.5], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 1], [-1, -1], [-1, -1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ValueError
    ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)



@pytest.mark.parametrize(
    "test, expected",
    [
        (['Terry Prachett'],['Terry Prachett']),
        (['Andy McNab'],['Andy McNab'])
    ])
def test_person(test, expected):
    """Test that a Person class is instantiated with the right attributes"""

    from inflammation.models import Person

    assert(Person(test).name == expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        (['Terry Prachett'], ['Terry Prachett']),
        (['Andy McNab'], ['Andy McNab'])
    ])
def test_patient(test, expected):
    """Test that a Person class is instantiated with the right attributes"""

    from inflammation.models import Patient

    assert (Patient(test).name == expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        (['Terry Prachett', 'Dr', []], ['Terry Prachett', 'Dr', []]),
        (['Andy McNab', 'Dr', []], ['Andy McNab', 'Dr', []])
    ])
def test_doctor(test, expected):
    """Test that a Person class is instantiated with the right attributes"""

    from inflammation.models import Doctor

    assert (Doctor(test[0]).name == expected[0])
    assert (Doctor(test[0]).title == expected[1])
    assert (Doctor(test[0]).patient_list == expected[2])


@pytest.mark.parametrize(
    "test",
    [
        (['Terry Prachett', 'Douglas Adams', 'Ursula Le Guin']),
    ])
def test_patient_group(test):
    """Test the instantiation of a group of patients"""

    from inflammation.models import PatientGroup, Patient

    patients = []
    for name in test:
        patients.append(Patient(name))

    assert (PatientGroup(patients).patients == patients)
    assert (type(PatientGroup(patients).control) == bool)