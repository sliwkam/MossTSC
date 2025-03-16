"""
This module provides functions to reduce the dimensions of 2D and 3D data arrays
using pooling and blockwise statistical operations. 
For 2D data, pooling operations are applied via `pool_2d_points` and blockwise
operations via `process_points_2D`. For 3D data, pooling operations are applied
via `pool_3d_points` and blockwise operations via `process_points_3D`.
All reduction methods require that the dimensions of the input arrays (excluding the
sample dimension) are divisible by a specified factor alpha.
"""

import numpy as np
from scipy.ndimage import zoom  


def check_divisibility_and_return(arr, alpha):
    """
    Check that all dimensions (excluding the first) of the input array are divisible by alpha.

    Parameters:
        arr (numpy.ndarray): input array
        alpha (int): divisor

    Returns:
        bool: true if all dimensions (ignoring the first) are divisible by alpha

    Raises:
        ValueError: if alpha is zero or if any dimension is not divisible by alpha
    """
    if alpha == 0:
        raise ValueError("Alpha cannot be zero.")
    for dim in arr.shape[1:]:
        if dim % alpha != 0:
            raise ValueError("Method not suitable. Wrong dimensions.")
    return True


def main_matrix_reduction_function(X_train, X_test, alpha, reduction_goal, method):
    """
    Reduce the dimensions of the train and test arrays using the specified reduction method.

    Supported methods:
        - 'pooling_sum'
        - 'pooling_std'
        - 'blockwise_sum'
        - 'blockwise_std'

    For 2D reduction:
        - 'pooling_sum' and 'pooling_std' use pool_2d_points
        - 'blockwise_sum' and 'blockwise_std' use process_points_2D

    For 3D reduction:
        - 'pooling_sum' and 'pooling_std' use pool_3d_points
        - 'blockwise_sum' and 'blockwise_std' use process_points_3D

    Parameters:
        X_train (numpy.ndarray): training data
        X_test (numpy.ndarray): test data
        alpha (int): divisibility factor
        reduction_goal (str): '2D' or '3D'
        method (str): reduction method (one of the supported methods)

    Returns:
        tuple:
            - reduced training data (numpy.ndarray)
            - reduced test data (numpy.ndarray)

    Raises:
        ValueError: if the method is not supported or if reduction_goal is invalid
    """
    valid_methods = ["pooling_sum", "pooling_std", "blockwise_sum", "blockwise_std"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}")

    # Check that array dimensions are divisible by alpha
    try:
        check_divisibility_and_return(X_train, alpha)
        check_divisibility_and_return(X_test, alpha)
    except ValueError as e:
        return str(e)

    if reduction_goal == '2D':
        if method == "pooling_sum":
            X_train_reduced = pool_2d_points(X_train, alpha, 'sum')
            X_test_reduced = pool_2d_points(X_test, alpha, 'sum')
        elif method == "pooling_std":
            X_train_reduced = pool_2d_points(X_train, alpha, 'std')
            X_test_reduced = pool_2d_points(X_test, alpha, 'std')
        elif method == "blockwise_sum":
            X_train_reduced = process_points_2D(X_train, alpha, 'sum')
            X_test_reduced = process_points_2D(X_test, alpha, 'sum')
        elif method == "blockwise_std":
            X_train_reduced = process_points_2D(X_train, alpha, 'std')
            X_test_reduced = process_points_2D(X_test, alpha, 'std')
    elif reduction_goal == '3D':
        if method == "pooling_sum":
            X_train_reduced = pool_3d_points(X_train, alpha, 'sum')
            X_test_reduced = pool_3d_points(X_test, alpha, 'sum')
        elif method == "pooling_std":
            X_train_reduced = pool_3d_points(X_train, alpha, 'std')
            X_test_reduced = pool_3d_points(X_test, alpha, 'std')
        elif method == "blockwise_sum":
            X_train_reduced = process_points_3D(X_train, alpha, 'sum')
            X_test_reduced = process_points_3D(X_test, alpha, 'sum')
        elif method == "blockwise_std":
            X_train_reduced = process_points_3D(X_train, alpha, 'std')
            X_test_reduced = process_points_3D(X_test, alpha, 'std')
    else:
        raise ValueError("Reduction goal must be '2D' or '3D'.")

    return X_train_reduced, X_test_reduced


def calculate_divisor(number, target):
    """
    Calculate the divisor by dividing the number by the target.

    Parameters:
        number (int): the number to be divided
        target (int): the target divisor

    Returns:
        int: the calculated divisor

    Raises:
        ValueError: if target is zero
    """
    if target == 0:
        raise ValueError("Target cannot be zero.")
    return int(number / target)


def pool_2d_points(input_array, alpha, operation):
    """
    Apply pooling operation on each 2D array using the specified operation.

    Supported operations: 'sum', 'std'.

    Parameters:
        input_array (numpy.ndarray): array of shape (n_points, rows, cols)
        alpha (int): factor to determine the pooling window size
        operation (str): pooling operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array of pooled 2D arrays

    Raises:
        ValueError: if an unsupported operation is provided
    """
    pool_size_row = calculate_divisor(input_array.shape[1], alpha)
    pool_size_col = calculate_divisor(input_array.shape[2], alpha)
    pooled_arrays = []
    for array in input_array:

        new_rows = array.shape[0] // pool_size_row
        # here we assume the first dimension of each 2D array corresponds to rows
        # however, if array is 2D, array.shape[0] is rows
        # to generalize, we can reshape based on row and column dimensions
        rows = array.shape[0]
        cols = array.shape[1]
        new_rows = rows // pool_size_row
        new_cols = cols // pool_size_col
        reshaped = array.reshape(new_rows, pool_size_row, new_cols, pool_size_col)
        if operation == 'sum':
            pooled_array = reshaped.sum(axis=(1, 3))
        elif operation == 'std':
            pooled_array = reshaped.std(axis=(1, 3))
        else:
            raise ValueError("Unsupported operation. Use 'sum' or 'std'.")
        pooled_arrays.append(pooled_array)
    return np.array(pooled_arrays)


def pool_3d_points(input_array, alpha, operation):
    """
    Apply pooling operation on each 3D array using the specified operation.

    Supported operations: 'sum', 'std'.

    Parameters:
        input_array (numpy.ndarray): array of shape (n_points, depth, rows, cols)
        alpha (int): factor to determine the pooling window size
        operation (str): pooling operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array of pooled 3D arrays

    Raises:
        ValueError: if an unsupported operation is provided
    """
    pool_size_depth = calculate_divisor(input_array.shape[1], alpha)
    pool_size_row = calculate_divisor(input_array.shape[2], alpha)
    pool_size_col = calculate_divisor(input_array.shape[3], alpha)
    
    pooled_arrays = []
    for array in input_array:
        new_depth = array.shape[0] // pool_size_depth
        new_rows = array.shape[1] // pool_size_row
        new_cols = array.shape[2] // pool_size_col
        reshaped = array.reshape(new_depth, pool_size_depth,
                                 new_rows, pool_size_row,
                                 new_cols, pool_size_col)
        if operation == 'sum':
            pooled_array = reshaped.sum(axis=(1, 3, 5))
        elif operation == 'std':
            pooled_array = reshaped.std(axis=(1, 3, 5))
        else:
            raise ValueError("Unsupported operation. Use 'sum' or 'std'.")
        pooled_arrays.append(pooled_array)
    return np.array(pooled_arrays)


def split_into_blocks_2d(array, block_size):
    """
    Split a 2D array into non-overlapping blocks of size block_size x block_size.

    Parameters:
        array (numpy.ndarray): 2D input array
        block_size (int): block size

    Returns:
        numpy.ndarray: array of blocks with shape (n_blocks, block_size, block_size)
    """
    num_blocks = [dim // block_size for dim in array.shape]
    reshaped = array.reshape(num_blocks[0], block_size, num_blocks[1], block_size)
    reshaped = reshaped.swapaxes(1, 2)
    return reshaped.reshape(-1, block_size, block_size)


def compute_statistics_block_2d(array, block_size, operation):
    """
    Compute a statistic on a 2D array by splitting it into blocks.

    Supported operations: 'sum', 'std'.

    Parameters:
        array (numpy.ndarray): 2D input array
        block_size (int): block size
        operation (str): statistical operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array representing the statistic computed on each block

    Raises:
        ValueError: if array dimensions are not divisible by block_size or if operation is unsupported
    """
    if array.shape[0] % block_size != 0 or array.shape[1] % block_size != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")
    blocks = split_into_blocks_2d(array, block_size)
    if operation == 'sum':
        result_block = np.sum(blocks, axis=0)
    elif operation == 'std':
        result_block = np.std(blocks, axis=0)
    else:
        raise ValueError("Unsupported operation. Use 'sum' or 'std'.")
    return result_block


def process_points_2D(original_array, block_size, operation):
    """
    Process each 2D array in the original array by computing blockwise statistics.

    Supported operations: 'sum', 'std'.

    Parameters:
        original_array (numpy.ndarray): array of shape (n_points, rows, cols)
        block_size (int): block size to split each 2D array
        operation (str): statistical operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array of processed 2D arrays
    """
    statistics_blocks = []
    for i in range(original_array.shape[0]):
        point_array = original_array[i]
        stats_block = compute_statistics_block_2d(point_array, block_size, operation)
        statistics_blocks.append(stats_block)
    return np.array(statistics_blocks)


def split_into_blocks_3d(array, block_size):
    """
    Split a 3D array into non-overlapping blocks of size block_size x block_size x block_size.

    Parameters:
        array (numpy.ndarray): 3D input array
        block_size (int): block size

    Returns:
        numpy.ndarray: array of blocks with shape (n_blocks, block_size, block_size, block_size)
    """
    num_blocks = [dim // block_size for dim in array.shape]
    reshaped = array.reshape(num_blocks[0], block_size,
                             num_blocks[1], block_size,
                             num_blocks[2], block_size)
    reshaped = reshaped.swapaxes(1, 2).swapaxes(3, 4)
    return reshaped.reshape(-1, block_size, block_size, block_size)


def compute_statistics_block_3d(array, block_size, operation):
    """
    Compute a statistic on a 3D array by splitting it into blocks.

    Supported operations: 'sum', 'std'.

    Parameters:
        array (numpy.ndarray): 3D input array
        block_size (int): block size
        operation (str): statistical operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array representing the statistic computed on each block.

    Raises:
        ValueError: if array dimensions are not divisible by block_size or if operation is unsupported
    """
    if array.shape[0] % block_size != 0 or array.shape[1] % block_size != 0 or array.shape[2] % block_size != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")
    blocks = split_into_blocks_3d(array, block_size)
    if operation == 'sum':
        result_block = np.sum(blocks, axis=0)
    elif operation == 'std':
        result_block = np.std(blocks, axis=0)
    else:
        raise ValueError("Unsupported operation. Use 'sum' or 'std'.")
    return result_block


def process_points_3D(original_array, block_size, operation):
    """
    Process each 3D array in the original array by computing blockwise statistics.

    Supported operations: 'sum', 'std'.

    Parameters:
        original_array (numpy.ndarray): array of shape (n_points, depth, rows, cols)
        block_size (int): block size to split each 3D array
        operation (str): statistical operation ('sum' or 'std')

    Returns:
        numpy.ndarray: array of processed 3D arrays
    """
    statistics_blocks = []
    for i in range(original_array.shape[0]):
        point_array = original_array[i]
        stats_block = compute_statistics_block_3d(point_array, block_size, operation)
        statistics_blocks.append(stats_block)
    return np.array(statistics_blocks)
