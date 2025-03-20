"""
This module provides functions to standardize training and test arrays using
ZScore standardization. The standardized values are computed using the mean
and standard deviation from the training data.
"""

import numpy as np


def main_standarization_function(X_train, X_test, standarization_method='ZScore'):
    """
    Standardize training and test arrays using ZScore standardization.

    The function computes the mean and standard deviation from the training data
    and applies ZScore standardization to both the training and test arrays.

    Parameters:
        X_train (numpy.ndarray): training data array
        X_test (numpy.ndarray): test data array

    Returns:
        tuple:
            - standardized_train (numpy.ndarray): standardized training data
            - standardized_test (numpy.ndarray): standardized test data
    """
    return z_score_standardizer_array(X_train, X_test)


def z_score_standardize_array(array, mean, std):
    """
    Standardize an array using the given mean and standard deviation.

    Parameters:
        array (numpy.ndarray): the array to standardize.
        mean (float): mean value for standardization.
        std (float): standard deviation for standardization.

    Returns:
        numpy.ndarray: the standardized array
    """
    return (array - mean) / std


def z_score_standardizer_array(train_data, test_data):
    """
    Standardize the train and test arrays using ZScore standardization.

    The mean and standard deviation are computed from the training data, and then
    used to standardize both the training and test arrays.

    Parameters:
        train_data (numpy.ndarray): training data array
        test_data (numpy.ndarray): test data array

    Returns:
        tuple:
            - standardized_train (numpy.ndarray): standardized training data
            - standardized_test (numpy.ndarray): standardized test data
    """
    mean = np.mean(train_data)
    std = np.std(train_data)
    
    standardized_train = z_score_standardize_array(train_data, mean, std)
    standardized_test = z_score_standardize_array(test_data, mean, std)
    
    return standardized_train, standardized_test