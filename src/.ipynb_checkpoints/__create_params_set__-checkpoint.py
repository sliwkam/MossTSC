import pandas as pd

# Define the parameter ranges
discretization_method = ['equal_width', 'equal_frequency', 'kmeans', 'entropy', 'hierarchical_clustering', 'max_entropy', 'pca', 'tree']
alpha = list(range(5, 21, 1))
beta = list(range(2, 8))
gamma = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 20]
ml_reduction = [False, True]
ml_reduction_method = ['variance', 'mean', 'std', 'max', 'sum', 'pca', 'umap','truncated_svd', 'nmf', 'fast_ica', 'factor_analysis', 'isomap', 'kernel_pca_linear', 'kernel_pca_poly', 'kernel_pca_rbf', 'kernel_pca_sigmoid', 'kernel_pca_cosine']
matrix_reduction_method = ["interpolation_linear", "interpolation_cubic", "interpolation_nearest", "pooling_mean", "pooling_max", "pooling_sum", "pooling_variance", "pooling_std", "downsampling", "blockwise_mean", "blockwise_max", "blockwise_sum", "blockwise_variance", "blockwise_std"]
standarization_method = ['MinMax', 'TFIDF', 'ZScore']
reduction_goal = ['2D', '3D']
learning_rate = [0.1, 0.01, 0.001, 0.0001]
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
sorted_df = df.sort_values(['alpha', 'beta', 'gamma', 'ml_reduction', 'discretization_method', 'ml_reduction_method', 'matrix_reduction_method', 'standarization_method', 'reduction_goal', 'learning_rate', 'model']).drop_duplicates().reset_index(drop=True)

# Save the sorted DataFrame to a CSV file
sorted_df.to_csv('ParamsToUse.csv', index=False)

print("ParamsToUse.csv has been created successfully.")