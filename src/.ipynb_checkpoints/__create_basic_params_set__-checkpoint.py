import pandas as pd
import os

output_dir = os.path.join('..', 'DataFiles')
output_file = 'ParamsToUseBasic.csv'

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the full path to the file
output_path = os.path.join(output_dir, output_file)

# Define the parameter ranges
discretization_method = ['equal_width', 'equal_frequency']
alpha = list(range(5, 21, 1))
beta = list(range(2, 8))
gamma = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 20]
ml_reduction = [False]
ml_reduction_method = ['variance']
matrix_reduction_method = ["interpolation_linear"]
standarization_method = ['ZScore']
reduction_goal = ['2D', '3D']
learning_rate = [0.001]
model = ['LeNet', 'ResNet']

# Initialize the list to hold all combinations
combinations = []

# Generate combinations based on the specified rules
for discret in discretization_method:
    for a in alpha:
        for b in beta:
            for g in gamma:
                for ml_red in ml_reduction:
                    for std_method in standarization_method:
                        for lr in learning_rate:
                            for mdl in model:
                                if b <= 3:
                                    ml_red = False
                                    ml_red_method = 'variance'
                                    mat_red_method = "interpolation_linear"
                                    red_goal = '2D'
                                    combinations.append((
                                        discret, a, b, g, ml_red, ml_red_method, mat_red_method, std_method, red_goal, lr, mdl
                                    ))
                                elif b > 3 and not ml_red:
                                    ml_red_method = 'variance'
                                    for mat_red_method in matrix_reduction_method:
                                        combinations.append((
                                            discret, a, b, g, ml_red, ml_red_method, mat_red_method, std_method, '2D', lr, mdl
                                        ))
                                        combinations.append((
                                            discret, a, b, g, ml_red, ml_red_method, mat_red_method, std_method, '3D', lr, mdl
                                        ))
                                elif b > 3 and ml_red:
                                    mat_red_method = "interpolation_linear"
                                    for ml_red_method in ml_reduction_method:
                                        combinations.append((
                                            discret, a, b, g, ml_red, ml_red_method, mat_red_method, std_method, '2D', lr, mdl
                                        ))
                                        combinations.append((
                                            discret, a, b, g, ml_red, ml_red_method, mat_red_method, std_method, '3D', lr, mdl
                                        ))

# Create a DataFrame from the combinations
df = pd.DataFrame(combinations, columns=[
    'discretization_method', 'alpha', 'beta', 'gamma', 'ml_reduction', 
    'ml_reduction_method', 'matrix_reduction_method', 'standarization_method', 
    'reduction_goal', 'learning_rate', 'model'
])

# Sort the DataFrame and drop duplicates
sorted_df = df.sort_values(['model', 'alpha', 'beta', 'gamma', 'ml_reduction', 'discretization_method', 'ml_reduction_method', 'matrix_reduction_method', 'standarization_method', 'reduction_goal', 'learning_rate']).drop_duplicates().reset_index(drop=True)

# Save the sorted DataFrame to a CSV file
sorted_df.to_csv(output_path, index=False)

print(f"{output_file} has been created successfully in {output_dir}.")
