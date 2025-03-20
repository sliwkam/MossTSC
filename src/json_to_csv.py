import os
import json
import pandas as pd

def process_json_files(folder_path, folder_name):
    data_list = []

    # Loop through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JSON file and contains the folder name
        if filename.endswith('.json') and folder_name in filename:
            file_path = os.path.join(folder_path, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

                # Check if 'data' is a list of dictionaries
                if isinstance(data, list):
                    for entry in data:
                        flattened_data = flatten_json(entry)
                        data_list.append(flattened_data)
                else:
                    flattened_data = flatten_json(data)
                    data_list.append(flattened_data)

    # Create a DataFrame from the list of flattened data
    df = pd.DataFrame(data_list)

    # Define the output CSV file path
    output_csv = os.path.join(folder_path, f'{folder_name}_summary.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"CSV file created: {output_csv}")

def flatten_json(data):
    """Function to flatten the JSON structure and rename keys"""
    return {
        'dataset_name': data['parameters']['dataset_name'],
        'discretization_method': data['parameters']['discretization_method'],
        'seed': data['parameters']['seed'],
        'alpha': data['parameters']['alpha'],
        'beta': data['parameters']['beta'],
        'gamma': data['parameters']['gamma'],
        'ml_reduction': data['parameters']['ml_reduction'],
        'ml_reduction_method': data['parameters']['ml_reduction_method'],
        'matrix_reduction_method': data['parameters']['matrix_reduction_method'],
        'standarization_method': data['parameters']['standarization_method'],
        'reduction_goal': data['parameters']['reduction_goal'],
        'learning_rate': data['parameters']['learning_rate'],
        'batch_size': data['parameters']['batch_size'],
        'model': data['parameters']['model'],
        'train_accuracy': data['metrics']['train']['accuracy'],
        'train_precision': data['metrics']['train']['precision'],
        'train_recall': data['metrics']['train']['recall'],
        'train_f1_score': data['metrics']['train']['f1_score'],
        'train_log_loss': data['metrics']['train']['log_loss'],
        'validation_accuracy': data['metrics']['validation']['accuracy'],
        'validation_precision': data['metrics']['validation']['precision'],
        'validation_recall': data['metrics']['validation']['recall'],
        'validation_f1_score': data['metrics']['validation']['f1_score'],
        'validation_log_loss': data['metrics']['validation']['log_loss'],
        'test_accuracy': data['metrics']['test']['accuracy'],
        'test_precision': data['metrics']['test']['precision'],
        'test_recall': data['metrics']['test']['recall'],
        'test_f1_score': data['metrics']['test']['f1_score'],
        'test_log_loss': data['metrics']['test']['log_loss'],
        'data_processing_elapsed_time': data['times']['data_processing_elapsed_time'],
        'model_elapsed_time': data['times']['model_elapsed_time'],
        'epoch_counter': data['epoch_counter'],
        'Error': data.get('Error', None)
    }

if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter the path to the folder containing JSON files: ")
    folder_name = input("Enter the folder name to filter JSON files: ")
    
    process_json_files(folder_path, folder_name)