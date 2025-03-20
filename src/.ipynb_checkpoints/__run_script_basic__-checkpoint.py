import pandas as pd
import concurrent.futures
import multiprocessing
import os
import json
import argparse
from datetime import datetime
import torch
import sys

# Import necessary functions from your other scripts
from _0_run_experiment import *
from _1a_34_DataProcessor_Utils import *

# Check for TPU availability
#try:
#    import torch_xla
#    import torch_xla.core.xla_model as xm
#    tpu_available = True
#    device = xm.xla_device()
#except ImportError:
#    tpu_available = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Custom JSON Encoder to handle non-serializable data types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

# Modified run_single_experiment function to use the detected device
def run_single_experiment(params, dataset_name, X, y, X_, y_, seed, batch_size):
    try:
        experiment_result = run_experiment(
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
            model=params['model'],
            epoch=1000,
            patience=125,
            device=device  # Pass the detected device
        )
        
        experiment_result['parameters']['dataset_name'] = dataset_name
        experiment_result['parameters']['seed'] = seed
        experiment_result['parameters']['batch_size'] = batch_size
        experiment_result['Error'] = None
        
    except Exception as e:
        experiment_result = {
            'parameters': params,
            'metrics': {
                'train': None,
                'validation': None,
                'test': None
            },
            'times': None,
            'epoch_counter': None,
            'Error': str(e)
        }
    
    return experiment_result

# Function to process a chunk of parameters
multiprocessing.set_start_method('spawn', force=True)

def process_chunk(chunk, dataset_name, X, y, X_, y_, seed, batch_size):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_experiment, row.to_dict(), dataset_name, X, y, X_, y_, seed, batch_size) for idx, row in chunk.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

# Function to save the last processed row
def save_last_row(row_number, results_dir, seed):
    last_row_file = os.path.join(results_dir, f'last_row_processed_{seed}.json')
    with open(last_row_file, 'w') as f:
        json.dump({'row_number': row_number}, f)

# Function to load the last processed row
def load_last_row(results_dir, seed):
    last_row_file = os.path.join(results_dir, f'last_row_processed_{seed}.json')
    if os.path.exists(last_row_file):
        with open(last_row_file, 'r') as f:
            data = json.load(f)
            return data.get('row_number', 0)
    return 0

def main(dataset_name, seed, batch_size):
    # Load the dataset
    X, y, X_, y_ = load_dataset(dataset_name)
    chunksize = 64
    
    # Define the directory paths
    datafiles_dir = os.path.join('..', 'DataFiles')
    params_file = os.path.join(datafiles_dir, 'ParamsToUse.csv')
    results_dir = os.path.join(datafiles_dir, f'{dataset_name}Results')
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load the last processed row
    last_row = load_last_row(results_dir, seed)
    chunk_number = 0

    # Read and process chunks of the ParamsToUse.csv file
    for chunk in pd.read_csv(params_file, chunksize=chunksize, skiprows=range(1, last_row + 1)):
        chunk_number += 1
        print(f"Processing chunk {chunk_number}")
        results = process_chunk(chunk, dataset_name, X, y, X_, y_, seed, batch_size)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'results_chunk_{dataset_name}_{seed}_{last_row}_{chunk_number}_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, cls=CustomJSONEncoder)

        # Update and save the last processed row
        last_row += len(chunk)
        save_last_row(last_row, results_dir, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with the specified dataset, seed, and batch size.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to be used.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for the experiment.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for the experiment.")

    args = parser.parse_args()

    main(args.dataset_name, args.seed, args.batch_size)
