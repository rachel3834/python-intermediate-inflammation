"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """
    Calculate the daily mean of a 2D inflammation data array.

    :param data: array
    :returns: mean of the array
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """
    Calculate the daily max of a 2D inflammation data array.

    :param data: array
    :returns: maximum value of the array
    """
    return np.max(data, axis=0)


def daily_min(data):
    """
    Calculate the daily min of a 2D inflammation data array.

    :param data: array
    :returns: minimum value of the array
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array."""
    print(data, type(data), data.dtype)
    if not isinstance(data, np.ndarray):
        raise TypeError('Inflammation data should be a numpy array')
    if not (data.dtype == np.int64):
        raise ValueError('Inflammation data should be a numpy array of integer values')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    if data.shape[1] != 3:
        raise ValueError('Inflammation data should have 3 columns')
    max_data = np.max(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised

