"""
This module provides functions for data discretization using two methods: equal width binning & equal frequency binning
Additional utility functions are provided to flatten data and map binned
numeric values to letters.
"""

import numpy as np


def flatten_data(X_train, X_test):
    """
    Flatten the training and test data along all dimensions except the last one.

    Parameters:
        X_train (numpy.ndarray): training data
        X_test (numpy.ndarray): test data

    Returns:
        tuple:
            - X_train_flat (numpy.ndarray): flattened training data
            - X_test_flat (numpy.ndarray): flattened test data
    """
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    return X_train_flat, X_test_flat


def map_to_letters(binned_data):
    """
    Map numeric binned data to corresponding letters of the alphabet.

    Parameters:
        binned_data (numpy.ndarray): array containing binned numeric data

    Returns:
        numpy.ndarray: array of the same shape as binned_data with numeric values
                       replaced by corresponding lowercase letters.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    letter_mapped_data = np.array([letters[int(b)] for b in binned_data.flatten()])
    return letter_mapped_data.reshape(binned_data.shape)


def perform_data_discretization(X_train, X_test, y_train=None, y_test=None,
                                method='equal_width', n_bins=5):
    """
    Discretize the training and test data using the specified method.
    
    Parameters:
        X_train (numpy.ndarray): training data to be discretized
        X_test (numpy.ndarray): test data to be discretized
        y_train (numpy.ndarray, optional): training labels (not used in these methods)
        y_test (numpy.ndarray, optional): test labels (not used in these methods)
        method (str): fiscretization method to use must be either 'equal_width' or 'equal_frequency'
        n_bins (int): number of bins to create

    Returns:
        tuple:
            - X_train_discretized (numpy.ndarray): discretized training data
            - X_test_discretized (numpy.ndarray): discretized test data

    Raises:
        ValueError: if an invalid method is specified
    """
    if method not in ['equal_width', 'equal_frequency']:
        raise ValueError("Method must be 'equal_width' or 'equal_frequency'.")

    if method == 'equal_width':
        X_train_disc, X_test_disc = apply_equal_width_binning(X_train, X_test, n_bins)
    elif method == 'equal_frequency':
        X_train_disc, X_test_disc = apply_equal_frequency_binning(X_train, X_test, n_bins)

    return X_train_disc, X_test_disc


def equal_width_binning_train(data, n_bins):
    """
    Train equal width binning on the data by computing bin edges.

    Parameters:
        data (numpy.ndarray): input data
        n_bins (int): number of bins to create

    Returns:
        numpy.ndarray: array of bin edges calculated from the input data
    """
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.linspace(min_val, max_val, n_bins + 1)
    return bins


def equal_width_binning_apply(data, bins):
    """
    Apply equal width binning to the data using precomputed bins.

    Parameters:
        data (numpy.ndarray): input data
        bins (numpy.ndarray): bin edges computed from the training data

    Returns:
        numpy.ndarray: binned data mapped to letters
    """
    binned_data = np.digitize(data, bins) - 1
    return map_to_letters(binned_data)


def apply_equal_width_binning(X_train, X_test, n_bins):
    """
    Apply equal width binning to both training and testing data.

    Parameters:
        X_train (numpy.ndarray): training data
        X_test (numpy.ndarray): test data
        n_bins (int): number of bins to create

    Returns:
        tuple:
            - X_train_eq_width (numpy.ndarray): discretized training data
            - X_test_eq_width (numpy.ndarray): discretized test data
    """
    bins_eq_width = equal_width_binning_train(X_train, n_bins)
    X_train_eq_width = equal_width_binning_apply(X_train, bins_eq_width)
    X_test_eq_width = equal_width_binning_apply(X_test, bins_eq_width)
    return X_train_eq_width, X_test_eq_width


def equal_frequency_binning_train(data, n_bins):
    """
    Train equal frequency binning on the data by computing bin edges based on quantiles.

    Parameters:
        data (numpy.ndarray): input data
        n_bins (int): number of bins to create

    Returns:
        numpy.ndarray: array of bin edges computed from quantiles of the data
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(data, quantiles)
    return bins


def equal_frequency_binning_apply(data, bins):
    """
    Apply equal frequency binning to the data using precomputed quantile-based bins.

    Parameters:
        data (numpy.ndarray): input data
        bins (numpy.ndarray): bin edges computed from the training data

    Returns:
        numpy.ndarray: binned data mapped to letters
    """
    binned_data = np.digitize(data, bins, right=True) - 1
    return map_to_letters(binned_data)


def apply_equal_frequency_binning(X_train, X_test, n_bins):
    """
    Apply equal frequency binning to both training and testing data.

    Parameters:
        X_train (numpy.ndarray): training data
        X_test (numpy.ndarray): test data
        n_bins (int): number of bins to create

    Returns:
        tuple:
            - X_train_eq_freq (numpy.ndarray): discretized training data
            - X_test_eq_freq (numpy.ndarray): discretized test data
    """
    bins_eq_freq = equal_frequency_binning_train(X_train, n_bins)
    X_train_eq_freq = equal_frequency_binning_apply(X_train, bins_eq_freq)
    X_test_eq_freq = equal_frequency_binning_apply(X_test, bins_eq_freq)
    return X_train_eq_freq, X_test_eq_freq
