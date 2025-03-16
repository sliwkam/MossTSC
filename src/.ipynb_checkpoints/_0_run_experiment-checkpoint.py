"""
This module defines the `run_experiment` function that orchestrates data processing,
model training, and evaluation for a classification task using deep learning models.
It integrates several data transformation and model training routines, iterating over
various configurations (e.g., dropout probabilities and weight decays) to obtain results.

Functions:
    run_experiment(dataset_name, X, y, X_, y_, discretization_method, alpha, beta, gamma,
                   statistical_reduction, statistical_reduction_method, matrix_reduction_method, standarization_method,
                   reduction_goal, seed, batch_size, learning_rate, epoch, patience, device)
"""

import os
import numpy as np
import time
from _8_data_transformation import *
from _1_DataProcessor_ import *
from _9a_models_run import *
from _99_pipeline_utils_ import *


def run_experiment(dataset_name, X, y, X_, y_, discretization_method, alpha, beta, gamma,
                   statistical_reduction, statistical_reduction_method, matrix_reduction_method, standarization_method,
                   reduction_goal, seed, batch_size, learning_rate, epoch, patience, device):
    """
    Run an experiment consisting of data processing and model training with multiple
    configurations, and return the resulting performance metrics and timing information.

    The experiment follows these steps:
      1. data processing: the input data (X, y, X_, y_) is resampled and transformed using a `DataTransformer` object. The transformed data is then reduced according to the specified machine learning and matrix reduction methods
      2. model training: for each combination of model ('AlexNet', 'LeNet'), dropout probability, and weight decay, the `modelling_main` function is called to train the model and record performance metrics
      3. error handling: if an error occurs during either data processing or model training, the error is captured and included in the output for the corresponding configuration

    Parameters:
        dataset_name (str): name of the dataset
        X (np.array or similar): training data features
        y (np.array or similar): training data labels
        X_ (np.array or similar): test data features
        y_ (np.array or similar): test data labels
        discretization_method (str): method used for discretizing the data
        alpha (int): parameter controlling discretization
        beta (int): parameter controlling discretization
        gamma (int): step size for joining symbols into words
        statistical_reduction (str): method for statistical reduction
        statistical_reduction_method (str): specific method for reducing training data
        matrix_reduction_method (str): method used for matrix reduction
        standarization_method (str): method used for data standardization
        reduction_goal (str): goal of the reduction step
        seed (int): random seed for reproducibility
        batch_size (int): batch size for model training
        learning_rate (float): learning rate for the optimizer
        epoch (int): maximum number of training epochs
        patience (int): patience for early stopping
        device (torch.device or str): device to use

    Returns:
        list of dict: a list of dictionaries where each dictionary contains:
            - 'parameters': a dict with the configuration used
            - 'metrics': a dict with training, validation, and test performance metrics
            - 'times': a dict with the elapsed times for data processing and model training
            - 'epoch_counter': the number of epochs run (or None if not available)
            - 'Error': error message if an exception occurred (or None otherwise)

    Notes:
        - the function iterates over multiple dropout probabilities (0, 0.3, 0.6) and weight decays (0, 0.1) for each model
        - if an error occurs during data processing, the function will generate a result for each model configuration with the corresponding error message
    """
    models = ['AlexNet', 'LeNet']
    dropout_probs = [0, 0.3, 0.6]
    weight_decays = [0, 0.1]
    results = []

    try:
        data_processing_start_time = time.time()
        transformer = DataTransformer()
        transformer.resample_train_and_test(X, y, X_, y_, seed)
        X_train, y_train, X_test, y_test = transformer.transform(
            alpha, beta, gamma, discretization_method=discretization_method, encode_labels=True
        )
        X_train_processed, X_test_processed = reduce_transform_data(
            X_train, X_test, beta, alpha, statistical_reduction=statistical_reduction,
            statistical_reduction_method=statistical_reduction_method, matrix_reduction_method=matrix_reduction_method,
            reduction_goal=reduction_goal, standarization_method=standarization_method
        )
        data_processing_end_time = time.time()
        data_processing_elapsed_time = data_processing_end_time - data_processing_start_time

        for dropout_prob in dropout_probs:
            for weight_decay in weight_decays:
                for model_name in models:
                    try:
                        model_start_time = time.time()
                        metrics_, epoch_counter = modelling_main(
                            X_train_processed, y_train, X_test_processed, y_test,
                            alpha=alpha, nn_model=model_name, num_epochs=epoch,
                            patience=patience, batch_size=batch_size,
                            learning_rate=learning_rate, device=device,
                            dropout_prob=dropout_prob, weight_decay=weight_decay
                        )
                        model_end_time = time.time()
                        model_elapsed_time = model_end_time - model_start_time

                        result = {
                            'parameters': {
                                'dataset_name': dataset_name,
                                'discretization_method': discretization_method,
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'statistical_reduction': statistical_reduction,
                                'statistical_reduction_method': statistical_reduction_method,
                                'matrix_reduction_method': matrix_reduction_method,
                                'standarization_method': standarization_method,
                                'reduction_goal': reduction_goal,
                                'seed': seed,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'model': model_name,
                                'epoch': epoch,
                                'patience': patience,
                                'dropout_prob': dropout_prob,
                                'weight_decay': weight_decay
                            },
                            'metrics': metrics_,
                            'times': {
                                'data_processing_elapsed_time': data_processing_elapsed_time,
                                'model_elapsed_time': model_elapsed_time
                            },
                            'epoch_counter': epoch_counter,
                            'Error': None
                        }
                    except Exception as model_e:
                        result = {
                            'parameters': {
                                'dataset_name': dataset_name,
                                'discretization_method': discretization_method,
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'statistical_reduction': statistical_reduction,
                                'statistical_reduction_method': statistical_reduction_method,
                                'matrix_reduction_method': matrix_reduction_method,
                                'standarization_method': standarization_method,
                                'reduction_goal': reduction_goal,
                                'seed': seed,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'model': model_name,
                                'epoch': epoch,
                                'patience': patience,
                                'dropout_prob': dropout_prob,
                                'weight_decay': weight_decay
                            },
                            'metrics': {
                                'train': None,
                                'validation': None,
                                'test': None
                            },
                            'times': {
                                'data_processing_elapsed_time': data_processing_elapsed_time,
                                'model_elapsed_time': None
                            },
                            'epoch_counter': None,
                            'Error': str(model_e)
                        }

                    results.append(result)

    except Exception as e:
        for model_name in models:
            for dropout_prob in dropout_probs:
                for weight_decay in weight_decays:
                    result = {
                        'parameters': {
                            'dataset_name': dataset_name,
                            'discretization_method': discretization_method,
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                            'statistical_reduction': statistical_reduction,
                            'statistical_reduction_method': statistical_reduction_method,
                            'matrix_reduction_method': matrix_reduction_method,
                            'standarization_method': standarization_method,
                            'reduction_goal': reduction_goal,
                            'seed': seed,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'model': model_name,
                            'epoch': epoch,
                            'patience': patience,
                            'dropout_prob': dropout_prob,
                            'weight_decay': weight_decay
                        },
                        'metrics': {
                            'train': None,
                            'validation': None,
                            'test': None
                        },
                        'times': {
                            'data_processing_elapsed_time': None,
                            'model_elapsed_time': None
                        },
                        'epoch_counter': None,
                        'Error': str(e)
                    }
                    results.append(result)

    return results