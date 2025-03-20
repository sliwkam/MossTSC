"""
This module provides a function to reduce and transform training and test data
using reshaping, statistical reduction, and standardization. Based on the provided
parameters, the function applies either statistical reduction or matrix reduction,
followed by data standardization.
"""

import os
import numpy as np
from _6_reshape_array import *
from _6a_statistical_reduction import *
from _6b_matrix_reduction import *
from _7_standarization import *


def reduce_transform_data(X_train, X_test, beta, alpha, statistical_reduction=False, statistical_reduction_method='sum',
                          matrix_reduction_method='pooling_sum', reduction_goal='2D',
                          standarization_method='ZScore'):
    """
    Reduce and transform the training and test data based on the specified parameters.

    The function applies a sequence of operations:
      1. if beta is less than or equal to 3, only standardization is applied
      2. otherwise, if machine learning reduction is enabled (statistical_reduction=True):
           - the data is flattened and sliced
           - the specified machine learning reduction method is applied
           - the reduced data is then reshaped back to a multi-dimensional array
           - finally, standardization is applied
      3. if machine learning reduction is not enabled:
           - the data is reshaped into 2D or 3D arrays (based on reduction_goal)
           - a matrix reduction method is applied
           - the resulting data is standardized

    Parameters:
        X_train (numpy.ndarray): training data array
        X_test (numpy.ndarray): test data array
        beta (int): parameter determining input data dimension. if beta <= 3, only standardization is applied
        alpha (int): parameter used for reshaping the data
        statistical_reduction (bool, optional): flag indicating if statistical reduction should be applied
        statistical_reduction_method (str, optional): method for statistical reduction
        matrix_reduction_method (str, optional): method for matrix reduction
        reduction_goal (str, optional): target dimension for reduction; either '2D' or '3D'
        standarization_method (str, optional): method for standardization

    Returns:
        tuple:
            - reshaped_train (numpy.ndarray): the transformed and reshaped training data
            - reshaped_test (numpy.ndarray): the transformed and reshaped test data
    """
    dim = int(reduction_goal[0])
    if beta <= 3:
        # if beta is less than or equal to 3, only standardize the train and test arrays
        return main_standarization_function(X_train, X_test, standarization_method=standarization_method)
    else:
        if statistical_reduction:
            # 1. flatten each example to 1D
            _1d_arrays_train = array_to_1d_cube_shape(X_train)
            _1d_arrays_test = array_to_1d_cube_shape(X_test)
            # 2. create 1D slices
            train_point, train_cell, train_slices = make_1d_slices(_1d_arrays_train, alpha, reduction_goal)
            test_point, test_cell, test_slices = make_1d_slices(_1d_arrays_test, alpha, reduction_goal)
            # 3. apply the specified statistical reduction method
            train_reduced, test_reduced = main_reduction_function(train_slices, test_slices, statistical_reduction_method)
            # 4. reshape the reduced data back into multi-dimensional arrays
            train_reduced_reshaped = reshape_column_for_each_point(train_point, train_cell, train_reduced, alpha, dim)
            test_reduced_reshaped = reshape_column_for_each_point(test_point, test_cell, test_reduced, alpha, dim)
            # 5. standardize the reshaped data
            return main_standarization_function(train_reduced_reshaped, test_reduced_reshaped, standarization_method=standarization_method)
        else:
            # apply matrix reduction:
            if reduction_goal == '2D':
                _arrays_train = array_to_2d_cube_shape(X_train, alpha)
                _arrays_test = array_to_2d_cube_shape(X_test, alpha)
            elif reduction_goal == '3D':
                _arrays_train = array_to_3d_cube_shape(X_train, alpha)
                _arrays_test = array_to_3d_cube_shape(X_test, alpha)
            else:
                raise ValueError("Invalid reduction_goal. Use '2D' or '3D'.")
            # apply the specified matrix reduction method
            train_reduced, test_reduced = main_matrix_reduction_function(_arrays_train, _arrays_test, alpha, reduction_goal, matrix_reduction_method)
            # standardize the reduced data
            return main_standarization_function(train_reduced, test_reduced, standarization_method=standarization_method)
