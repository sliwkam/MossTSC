"""
This module provides functions for discretizing numerical time series into
symbolic representations (letters), joining symbols into words, creating
count matrices for words, normalizing arrays, and loading classification
datasets. 

Functions:
    discretize_serie(arr, alpha, global_min, global_max)
    discretize_arrays(train_arrays, test_arrays, alpha)
    discretize_serie_equal_frequency(arr, alpha)
    discretize_arrays_equal_frequency(train_arrays, test_arrays, alpha)
    join_symbols_to_words(symbols_arrays, beta, gamma)
    create_count_matrix(alpha, beta, values_arrays)
    normalize_array(arr, arr2)
    load_dataset(name, extract_path="./Temp/")
"""

import numpy as np
from aeon.datasets import load_classification


def discretize_serie(arr, alpha, global_min, global_max):
    """
    Discretize the values in a NumPy array into letters.

    This function maps the continuous values in `arr` to a set of letters
    by splitting the interval [global_min, global_max] into `alpha` bins.

    Parameters:
        arr (np.array): Input array to be discretized.
        alpha (int): Number of distinct letters to discretize into. Must be
                     a positive integer not exceeding 26.
        global_min (float): Minimum value for the discretization bins.
        global_max (float): Maximum value for the discretization bins.

    Returns:
        np.array: An array of the same shape as `arr` containing the mapped
                  letter values.

    Raises:
        ValueError: If `alpha` is less than 1, is not an integer, or exceeds 26.
    """
    if alpha < 1 or not isinstance(alpha, int):
        raise ValueError("alpha must be a positive integer")
    if alpha > 26:
        raise ValueError("maximum allowed letters is 26")
    
    letters = [chr(i) for i in range(97, 97 + alpha)]
    bins = np.linspace(global_min, global_max, num=alpha + 1, endpoint=True)[:-1]
    digitized = np.digitize(arr, bins, right=True)
    discrete_arr = np.vectorize(lambda x: letters[x - 1])(digitized)
    
    return discrete_arr


def discretize_arrays(train_arrays, test_arrays, alpha):
    """
    Discretize all arrays in train and test sets into letters.

    The function computes the global minimum and maximum from the training arrays
    and then discretizes each array in both train and test datasets using these
    bounds.

    Parameters:
        train_arrays (list of np.array): List of training arrays.
        test_arrays (list of np.array): List of test arrays.
        alpha (int): Number of letters to discretize into.

    Returns:
        tuple:
            - list of np.array: Discretized training arrays.
            - list of np.array: Discretized test arrays.
    """
    global_min = np.min([np.min(arr) for arr in train_arrays])
    global_max = np.max([np.max(arr) for arr in train_arrays])
    
    discretized_train = [
        discretize_serie(arr, alpha, global_min, global_max) for arr in train_arrays
    ]
    discretized_test = [
        discretize_serie(arr, alpha, global_min, global_max) for arr in test_arrays
    ]
    
    return discretized_train, discretized_test


def discretize_serie_equal_frequency(arr, alpha):
    """
    Discretize the values in a NumPy array into letters using equal frequency bins.

    The discretization is based on quantiles so that each interval contains an equal
    number of observations. If duplicate quantiles reduce the number of bins, the number
    of letters is adjusted accordingly.

    Parameters:
        arr (np.array): Input array to be discretized.
        alpha (int): Desired number of letters to discretize into.

    Returns:
        np.array: An array with the discretized letter values.

    Raises:
        ValueError: If `alpha` is less than 1, is not an integer, or exceeds 26.
    """
    if alpha < 1 or not isinstance(alpha, int):
        raise ValueError("alpha must be a positive integer")
    if alpha > 26:
        raise ValueError("maximum allowed letters is 26")
    
    sorted_arr = np.sort(arr.flatten())
    quantiles = [np.percentile(sorted_arr, q) for q in np.linspace(0, 100, alpha + 1)]
    unique_quantiles = np.unique(quantiles)
    letters = [chr(i) for i in range(97, 97 + min(alpha, len(unique_quantiles) - 1))]
    
    digitized = np.digitize(arr, unique_quantiles, right=True)
    discrete_arr = np.vectorize(
        lambda x: letters[min(x - 1, len(letters) - 1)]
    )(digitized)
    
    return discrete_arr


def discretize_arrays_equal_frequency(train_arrays, test_arrays, alpha):
    """
    Discretize all arrays in train and test sets into letters using equal frequency bins.

    The function uses the quantiles derived from the combined training data to ensure
    that each bin contains an equal number of observations.

    Parameters:
        train_arrays (list of np.array): List of training arrays.
        test_arrays (list of np.array): List of test arrays.
        alpha (int): Desired number of letters for discretization.

    Returns:
        tuple:
            - list of np.array: Discretized training arrays.
            - list of np.array: Discretized test arrays.
    """
    combined_train_data = np.concatenate(train_arrays)
    # Although combined_train_data is computed here, each array is discretized independently.
    discretized_train = [
        discretize_serie_equal_frequency(arr, alpha) for arr in train_arrays
    ]
    discretized_test = [
        discretize_serie_equal_frequency(arr, alpha) for arr in test_arrays
    ]
    
    return discretized_train, discretized_test


def join_symbols_to_words(symbols_arrays, beta, gamma):
    """
    Join symbols to form words for each array of symbols.

    For each 1D array of symbols (flattened from potentially 2D arrays), the
    function concatenates contiguous segments of length `beta` into words, moving
    forward `gamma` symbols between the start of consecutive words.

    Parameters:
        symbols_arrays (list of np.array): List of arrays containing symbols.
        beta (int): Desired length of each word.
        gamma (int): Step size between the starting indices of consecutive words.

    Returns:
        np.array: An array of arrays, where each sub-array contains the words
                  created from the corresponding symbols array.

    Raises:
        ValueError: If `beta` or `gamma` is not a positive integer.
    """
    if beta <= 0 or gamma <= 0:
        raise ValueError("Both beta and gamma must be positive integers.")
    
    all_words = []
    for symbols_array in symbols_arrays:
        symbols = symbols_array.ravel()
        words = []
        for i in range(0, len(symbols) - beta + 1, gamma):
            if len(symbols[i:i + beta]) < beta:
                break
            word = ''.join(symbols[i:i + beta])
            words.append(word)
        all_words.append(np.array(words))
    
    return np.array(all_words)


def create_count_matrix(alpha, beta, values_arrays):
    """
    Create count matrices for word arrays.

    For each sublist of words in `values_arrays`, the function builds an N-dimensional
    count matrix (with N equal to `beta`) where each dimension has a size equal to `alpha`.
    Each cell in the matrix corresponds to the frequency count of the word represented
    by the specific combination of letters.

    Parameters:
        alpha (int): Number of possible letters.
        beta (int): Length of words to be considered.
        values_arrays (list of list of str): A list where each element is a list of words.

    Returns:
        np.ndarray: An array of count matrices, one for each sublist in `values_arrays`.
    """
    letters = [chr(i) for i in range(97, 97 + alpha)]
    count_matrices = []
    
    for values in values_arrays:
        matrix = np.zeros([alpha] * beta, dtype=np.int16)
        for value in values:
            if len(value) == beta and all(c in letters for c in value):
                indices = [letters.index(c) for c in value]
                matrix[tuple(indices)] += 1
        count_matrices.append(matrix)
    
    return np.array(count_matrices)


def normalize_array(arr, arr2):
    """
    Normalize two nested arrays so that all values lie between 0 and 1.

    The function computes the minimum and maximum values from the first nested array (`arr`)
    and applies min-max normalization to both `arr` and `arr2`.

    Parameters:
        arr (list of list of float/int): Nested array (typically training data).
        arr2 (list of list of float/int): Nested array (typically test data).

    Returns:
        tuple:
            - np.array: Normalized version of `arr`.
            - np.array: Normalized version of `arr2`.
    """
    arr_np = np.array(arr)
    arr_np2 = np.array(arr2)
    min_val = arr_np.min()
    max_val = arr_np.max()
    normalized_arr = (arr_np - min_val) / (max_val - min_val)
    normalized_arr2 = (arr_np2 - min_val) / (max_val - min_val)
    
    return normalized_arr, normalized_arr2


def load_dataset(name, extract_path="./Temp/"):
    """
    Load a classification dataset and split it into training and test sets.

    The function utilizes the `load_classification` method from the `aeon.datasets`
    package to load and extract the specified dataset.

    Parameters:
        name (str): The name of the dataset (e.g., "ElectricDevices").
        extract_path (str): Directory path for data extraction. Default is "./Temp/".

    Returns:
        tuple:
            - X_train (np.array): Training data.
            - y_train (np.array): Labels for the training data.
            - X_test (np.array): Test data.
            - y_test (np.array): Labels for the test data.
    """
    X_train, y_train = load_classification(
        name, extract_path=extract_path, split="TRAIN", return_metadata=False
    )
    X_test, y_test = load_classification(
        name, extract_path=extract_path, split="test", return_metadata=False
    )
    
    return X_train, y_train, X_test, y_test
