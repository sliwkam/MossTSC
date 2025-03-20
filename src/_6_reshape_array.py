"""
This module provides functions for reshaping multi-dimensional arrays into 
1D, 2D, or 3D cube-like shapes. The reshaping functions attempt to create 
shapes that are best suited for further processing.
"""

import numpy as np
import pandas as pd


def array_to_1d_cube_shape(arr):
    """
    Flatten each example in the array to a 1D vector.

    This function reshapes the input array such that each sample (the first dimension)
    is flattened into a single dimension, while preserving the number of samples.

    Parameters:
        arr (numpy.ndarray): input array with shape (n_samples, ...)

    Returns:
        numpy.ndarray: reshaped array with shape (n_samples, n_features)
    """
    return arr.reshape(arr.shape[0], -1)


def array_to_2d_cube_shape(arr, alpha=1):
    """
    Reshape each example in the array to a 2D matrix.

    The function attempts to reshape each sample into the best fitting rectangular shape.
    It adjusts the total number of elements to be divisible by `alpha` without a remainder.
    If a square shape is possible, the sample is reshaped into a square; otherwise, the 
    function iterates to find a suitable pair of dimensions.

    Parameters:
        arr (numpy.ndarray): input array with shape (n_samples, ...)
        alpha (int, optional): divisibility factor for the dimensions (default is 1)

    Returns:
        numpy.ndarray: reshaped array with shape (n_samples, rows, columns)

    Raises:
        ValueError: if a suitable 2D shape cannot be determined
    """
    other_dims = np.prod(arr.shape[1:])
 
    if other_dims % alpha != 0:
        other_dims += alpha - (other_dims % alpha)
    side_length = int(np.sqrt(other_dims))

    if side_length * side_length == other_dims:
        return arr.reshape(arr.shape[0], side_length, side_length)
    else:
        for i in range(side_length, 0, -1):
            if other_dims % i == 0 and i % alpha == 0 and (other_dims // i) % alpha == 0:
                return arr.reshape(arr.shape[0], i, other_dims // i)
    raise ValueError("Cannot reshape array to a 2D cube with the given alpha value.")


def array_to_3d_cube_shape(arr, alpha=1):
    """
    Reshape each example in the array to a 3D cube-like shape.

    The function reshapes each sample into three dimensions that are as balanced as possible.
    It adjusts the total number of elements to be divisible by `alpha` and searches for the 
    dimensions that minimize the difference between them while ensuring each is divisible by alpha.

    Parameters:
        arr (numpy.ndarray): input array with shape (n_samples, ...)
        alpha (int, optional): divisibility factor for the dimensions (default is 1)

    Returns:
        numpy.ndarray: reshaped array with shape (n_samples, dim1, dim2, dim3)
    """
    total_elements = np.prod(arr.shape[1:])
    if total_elements % alpha != 0:
        total_elements += alpha - (total_elements % alpha)
        
    best_diff = float('inf')
    best_dims = (alpha, alpha, total_elements // (alpha * alpha))

    for i in range(alpha, int(total_elements**(1/3)) + 1, alpha):
        if total_elements % i == 0:
            remaining = total_elements // i
            for j in range(alpha, int(remaining**(1/2)) + 1, alpha):
                if remaining % j == 0:
                    k = remaining // j
                    diff = abs(i - j) + abs(j - k) + abs(k - i)
                    if diff < best_diff and i % alpha == 0 and j % alpha == 0 and k % alpha == 0:
                        best_diff = diff
                        best_dims = (i, j, k)
    
    dim1, dim2, dim3 = best_dims
    return arr.reshape(arr.shape[0], dim1, dim2, dim3)
