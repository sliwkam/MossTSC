"""
This module provides functions to run a single experiment for deep learning models
using specified parameters and datasets. It includes a custom JSON encoder to handle
non-serializable data types (e.g., pandas objects and PyTorch tensors) and a function
to run an experiment with given hyperparameters, data, and configurations.

Functions:
    run_single_experiment(params, dataset_name, X, y, X_, y_, seed, batch_size, device)
Classes:
    CustomJSONEncoder
"""

import pandas as pd
import torch
import json

from _0_run_experiment import *
from discretization_module_utils import *  


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder that extends the default JSONEncoder to support additional data types.

    This encoder converts pandas Series and DataFrames to dictionaries and PyTorch tensors
    to lists, making them serializable to JSON.

    Methods:
        default(obj): Returns a JSON serializable representation for non-standard types.
    """
    def default(self, obj):
        """
        Provide a JSON serializable representation for non-standard data types.

        Parameters:
            obj: The object to encode.

        Returns:
            A JSON serializable representation of the object.

        Notes:
            - pandas.Series and pandas.DataFrame are converted to dictionaries.
            - torch.Tensor is converted to a list.
            - Other types are handled by the superclass.
        """
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

def run_single_experiment(params, dataset_name, X, y, X_, y_, seed, batch_size, device):
    """
    Run a single experiment for the specified dataset and deep learning models.

    This function transfers data to the specified device, runs experiments using the
    provided parameters and configurations (such as discretization and reduction methods),
    and returns the results for each model. If an error occurs during the experiment,
    the function returns a result with the error message for each model.

    Parameters:
        params (dict): Dictionary containing experiment hyperparameters and configurations.
        dataset_name (str): Name of the dataset being used.
        X (torch.Tensor or other): Training data features.
        y (torch.Tensor or other): Training data labels.
        X_ (torch.Tensor or other): Test data features.
        y_ (torch.Tensor or other): Test data labels.
        seed (int): Seed value for reproducibility.
        batch_size (int): Batch size for training.
        device (torch.device or str): Device (e.g., 'cpu' or 'cuda') to run the experiment on.

    Returns:
        list of dict: A list containing dictionaries with experiment results for each model.
                      Each dictionary includes parameters, performance metrics (train, validation,
                      and test), timing information, epoch counter, and error message (if any).

    Raises:
        Exception: Any exceptions raised during the experiment are caught and included in the
                   results dictionary under the 'Error' key.
    """
    models = ['AlexNet', 'LeNet']
    try:
        # Transfer data to the appropriate device
        if isinstance(X, torch.Tensor):
            X, y = X.to(device), y.to(device)
        if isinstance(X_, torch.Tensor):
            X_, y_ = X_.to(device), y_.to(device)

        experiment_results = run_experiment(
            dataset_name=dataset_name,
            X=X,
            y=y,
            X_=X_,
            y_=y_,
            discretization_method=params['discretization_method'],
            alpha=params['alpha'],
            beta=params['beta'],
            gamma=params['gamma'],
            ml_reduction=params['ml_reduction'],
            ml_reduction_method=params['ml_reduction_method'],
            matrix_reduction_method=params['matrix_reduction_method'],
            standarization_method=params['standarization_method'],
            reduction_goal=params['reduction_goal'],
            seed=seed,
            batch_size=batch_size,
            learning_rate=params['learning_rate'],
            epoch=1000,
            patience=75,
            device=device
        )

        # Append dataset-specific parameters to both results
        for result in experiment_results:
            result['parameters']['dataset_name'] = dataset_name
            result['parameters']['seed'] = seed
            result['parameters']['batch_size'] = batch_size

    except Exception as e:
        # If any exception occurs during the experiment, create a result with the error message
        experiment_results = [{
            'parameters': params,
            'metrics': {
                'train': None,
                'validation': None,
                'test': None
            },
            'times': None,
            'epoch_counter': None,
            'Error': str(e)
        } for _ in models]

    return experiment_results