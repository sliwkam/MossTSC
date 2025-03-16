"""
This module defines the DataTransformer class, which performs data resampling,
discretization, and transformation for classification experiments. It includes methods
to resample training and test data based on class distribution and to transform the data
by discretizing, joining symbols into words, and creating count matrices. Optionally, label
encoding can be applied.
"""

from discretization_module_utils import *
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.utils import shuffle, check_random_state
import pandas as pd

from _2_discretization import *


class DataTransformer:
    """
    A transformer class to resample and transform time series data for classification tasks.

    The DataTransformer handles:
      - resampling training and test data based on class distributions
      - discretizing time series data into symbolic representations
      - joining discretized symbols into words
      - creating count matrices from word arrays
      - optionally encoding labels

    Attributes:
        X_train (numpy.ndarray): the training data after resampling
        y_train (numpy.ndarray): the training labels after resampling
        X_test (numpy.ndarray): the test data after resampling
        y_test (numpy.ndarray): the test labels after resampling
        X_train_out (numpy.ndarray): transformed training data (count matrices)
        y_train_out (numpy.ndarray): transformed training labels
        X_test_out (numpy.ndarray): transformed test data (count matrices)
        y_test_out (numpy.ndarray): transformed test labels
    """

    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def resample_train_and_test(self, X_train, y_train, X_test, y_test, seed):
        """
        Resample training and test data to ensure balanced class distributions.

        This method first flattens the training and test data, and then, using the given
        seed, shuffles and splits the concatenated data based on class labels. If the seed
        is zero, no resampling is performed and the original data is stored.

        Parameters:
            X_train (numpy.ndarray): training data
            y_train (numpy.ndarray): training labels
            X_test (numpy.ndarray): test data
            y_test (numpy.ndarray): test labels
            seed (int): random seed for reproducibility

        Raises:
            ValueError: if there are insufficient examples for any class to perform the split

        Returns:
            None
        """
        np.random.seed(seed)
        X_train, X_test = flatten_data(X_train, X_test)

        if seed == 0:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            return

        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)

        classes = np.unique(y_all)
        train_indices = []
        test_indices = []

        for cls in classes:
            cls_indices = np.where(y_all == cls)[0]
            np.random.shuffle(cls_indices)
            n_cls_train = np.sum(y_train == cls)
            n_cls_test = np.sum(y_test == cls)
            total_required = n_cls_train + n_cls_test
            total_available = len(cls_indices)
            if total_required > total_available:
                raise ValueError(f"Not enough examples for class {cls} to perform split.")
            train_indices_cls = cls_indices[:n_cls_train]
            test_indices_cls = cls_indices[n_cls_train:n_cls_train + n_cls_test]
            train_indices.extend(train_indices_cls)
            test_indices.extend(test_indices_cls)

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        X_train_new = X_all[train_indices]
        y_train_new = y_all[train_indices]
        X_test_new = X_all[test_indices]
        y_test_new = y_all[test_indices]

        self.X_train = np.array(X_train_new)
        self.y_train = np.array(y_train_new)
        self.X_test = np.array(X_test_new)
        self.y_test = np.array(y_test_new)

        return

    def transform(self, alpha, beta, gamma, discretization_method='equal_width', encode_labels=True):
        """
        Transform the resampled data by discretizing, forming words, and creating count matrices.

        The transformation process involves:
          1. discretizing the training and test arrays into symbolic representations using the specified discretization method
          2. joining the symbols into words of length beta with a step gamma
          3. creating count matrices from the word arrays
          4. optionally, encoding the labels using LabelEncoder

        Parameters:
            alpha (int): parameter for discretization (number of bins)
            beta (int): length of words to be formed
            gamma (int): step size for joining symbols into words
            discretization_method (str, optional): discretization method ('equal_width' by default)
            encode_labels (bool, optional): flag to indicate whether labels should be encoded. Default is True

        Returns:
            tuple:
                - matrixes_train (numpy.ndarray): count matrices for training data
                - y_train (numpy.ndarray): encoded training labels
                - matrixes_test (numpy.ndarray): count matrices for test data
                - y_test (numpy.ndarray): encoded test labels
        """
        # Discretize training and test data into symbolic representations.
        disc_train_x, disc_test_x = perform_data_discretization(
            X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test,
            method=discretization_method, n_bins=alpha
        )

        # Join symbols to form words with the specified length (beta) and step (gamma).
        words_train = join_symbols_to_words(disc_train_x, beta, gamma)
        words_test = join_symbols_to_words(disc_test_x, beta, gamma)

        # Create count matrices from the word arrays.
        matrixes_train = create_count_matrix(alpha, beta, words_train)
        matrixes_test = create_count_matrix(alpha, beta, words_test)

        # Encode labels if required.
        if encode_labels:
            le = LabelEncoder().fit(self.y_train)
            y_train = le.transform(self.y_train)
            y_test = le.transform(self.y_test)
        else:
            y_train = self.y_train
            y_test = self.y_test

        self.X_train_out = matrixes_train
        self.y_train_out = y_train
        self.X_test_out = matrixes_test
        self.y_test_out = y_test

        return matrixes_train, y_train, matrixes_test, y_test
