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
    if not (data.dtype == np.int64 or data.dtype == np.int32):
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


class Observation:
    def __init__(self, day, value):
        self.day = day
        self.value = value

    def __str__(self):
        return 'Day '+str(self.day)+': '+str(self.value)

    def __eq__(self, observation2):
        if (self.day == observation2.day
            and self.value == observation2.value):
            return True
        else:
            return False

class Person:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Patient(Person):
    def __init__(self, name, observations=None):
        super().__init__(name)
        self.observations = []

        if observations is not None:
            self.observations = observations

    def add_observation(self, value, day=None):
        if day is None:
            if len(self.observations) > 0:
                day = self.observations[-1].day + 1
            else:
                day = 0

        new_obs = Observation(day, value)

        self.observations.append(new_obs)

        return new_obs

    def __eq__(self, patient2):
        if self.name == patient2.name:
            same_obs = []
            for datum1,datum2 in zip(self.observations, patient2.observations):
                same_obs.append( (datum1 == datum2) )
            if False in same_obs:
                return False
            else:
                return True
        else:
            return False


class Doctor(Person):
    def __init__(self, name):
        super().__init__(name)
        self.title = 'Dr'
        self.patient_list = []

    def __str__(self):
        return self.title+'.'+self.name


class PatientGroup:
    def __init__(self, patients=[]):
        self.patients = patients
        self.control = False
