"""
This module provides utility functions for reshaping data and performing
dimensionality reduction using two statistical methods: sum and standard
deviation. The functions include slicing a 1D array into 2D slices, reshaping
values for each unique point.
"""

import numpy as np
from _6_reshape_array import *
from _99_pipeline_utils_ import *


def make_1d_slices(data, alpha, target_dim='2D'):
    """
    Reshape a 1D array per example into a 2D array based on the given alpha.

    Parameters:
        data (numpy.ndarray): the input array where each example is 1D
        alpha (int): the alpha value used for determining the new dimensions
        target_dim (str, optional): target dimension, either '2D' or '3D' (default is '2D')

    Returns:
        tuple:
            - points (numpy.ndarray): array of repeated indices for each example
            - cells (numpy.ndarray): array of tiled indices for subarrays within each example
            - reshaped_2d_array (numpy.ndarray): the reshaped 2D array

    Raises:
        ValueError: if target_dim is not '2D' or '3D'
    """
    if target_dim == '2D':
        length = get_divider(data.shape[1], alpha ** 2)
    elif target_dim == '3D':
        length = get_divider(data.shape[1], alpha ** 3)
    else:
        raise ValueError("Invalid target_dim. Use '2D' or '3D'.")

    num_subarrays = data.shape[1] // length
    reshaped_2d_array = data.reshape(-1, length)
    points = np.repeat(np.arange(data.shape[0]), num_subarrays)
    cells = np.tile(np.arange(num_subarrays), data.shape[0])
    
    return points, cells, reshaped_2d_array


def reshape_column_for_each_point(points, cells, values, alpha, dim):
    """
    Reshape values associated with each unique point into a multidimensional array.

    Parameters:
        points (numpy.ndarray): array of points
        cells (numpy.ndarray): array of cells (not used in reshaping but assumed to be associated with points)
        values (numpy.ndarray): values corresponding to the points
        alpha (int): the size of each dimension after reshaping
        dim (int): the number of dimensions for the reshaped array

    Returns:
        numpy.ndarray: an array of reshaped arrays for each unique point
    """
    unique_points = np.unique(points)
    reshaped_arrays = []

    for point in unique_points:
        subset = values[points == point]
        reshaped_array = subset.reshape((alpha,) * dim)
        reshaped_arrays.append(reshaped_array)

    return np.array(reshaped_arrays)


def main_reduction_function(train_array, test_array, method):
    """
    Reduce the dimensions of the train and test arrays using the specified method.

    This main reduction function supports only two methods:
      - 'std': computes the standard deviation along axis 1
      - 'sum': computes the sum along axis 1

    Parameters:
        train_array (numpy.ndarray): the training data array
        test_array (numpy.ndarray): the test data array
        method (str): the reduction method to use; must be either 'std' or 'sum'

    Returns:
        tuple:
            - reduced training data (numpy.ndarray)
            - reduced test data (numpy.ndarray)

    Raises:
        ValueError: if an unknown method is specified
    """
    if method == 'std':
        return std_reduction(train_array, test_array)
    elif method == 'sum':
        return sum_reduction(train_array, test_array)
    else:
        raise ValueError(f"Unknown method: {method}")


def std_reduction(train_array, test_array):
    """
    Reduce the train and test arrays by computing the standard deviation along axis 1.

    Parameters:
        train_array (numpy.ndarray): the training data array
        test_array (numpy.ndarray): the test data array

    Returns:
        tuple:
            - standard deviation of training data (numpy.ndarray)
            - standard deviation of test data (numpy.ndarray)
    """
    return np.std(train_array, axis=1), np.std(test_array, axis=1)


def sum_reduction(train_array, test_array):
    """
    Reduce the train and test arrays by computing the sum along axis 1.

    Parameters:
        train_array (numpy.ndarray): the training data array
        test_array (numpy.ndarray): the test data array

    Returns:
        tuple:
            - sum of training data (numpy.ndarray)
            - sum of test data (numpy.ndarray)
    """
    return np.sum(train_array, axis=1), np.sum(test_array, axis=1)
