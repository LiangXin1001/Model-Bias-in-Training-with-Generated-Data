import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str, nargs='+', required=True, help='The names of the models to process results for')
args = parser.parse_args()

base_path = 'results'
subdirectories = [str(i) for i in range(11)]

# Function to read TPR and FPR files
def read_tpr_fpr_files(base_path, subdirectories, model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name,f"gen{subdir}", f'tpr_fpr_results.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Gen'] = subdir  # Add generation identifier
            dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df
colors = ['blue', 'green', 'red', 'purple', 'orange']   
color_index = 0
# Initialize plot for TPR and FPR across models
plt.figure(figsize=(15, 6))
for model_name in args.model_names:
    combined_df = read_tpr_fpr_files(base_path, subdirectories, model_name)    
    # Calculate average TPR and FPR per generation
    average_tpr_per_gen = combined_df.groupby('Gen')['TPR'].mean()
    # Ensure the 'Gen' index is numeric for sorting
    average_tpr_per_gen.index = average_tpr_per_gen.index.astype(int)
    # Sort by generation
    average_tpr_per_gen = average_tpr_per_gen.sort_index()
    # Plot TPR and FPR for each model
    plt.plot(average_tpr_per_gen.index, average_tpr_per_gen.values,color=colors[color_index], marker='o', label=f'TPR - {model_name}')
    color_index = color_index + 1
plt.xlabel('Generation')
plt.ylabel('Value')
plt.title('Average TPR Values Across Generations for Multiple Models')
plt.xticks(average_tpr_per_gen.index)
plt.ylim(0.9, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the combined plot
output_path = 'images/all/combined_tpr_plot.png'
plt.savefig(output_path)
plt.close()
 
print(f'Combined TPR  plot saved as {output_path}')


color_index = 0
plt.figure(figsize=(15, 6))
for model_name in args.model_names:
    combined_df = read_tpr_fpr_files(base_path, subdirectories, model_name)
    average_fpr_per_gen = combined_df.groupby('Gen')['FPR'].mean()
    average_fpr_per_gen.index = average_fpr_per_gen.index.astype(int)
    average_fpr_per_gen = average_fpr_per_gen.sort_index()
    plt.plot(average_fpr_per_gen.index, average_fpr_per_gen.values,color=colors[color_index], marker='x', label=f'FPR - {model_name}')
    color_index = color_index + 1
plt.xlabel('Generation')
plt.ylabel('Value')
plt.title('Average FPR Values Across Generations for Multiple Models')
plt.xticks(average_fpr_per_gen.index)
plt.ylim(0, 0.1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the combined plot
output_path = 'images/all/combined_fpr_plot.png'
plt.savefig(output_path)
plt.close()
 
print(f'Combined FPR plot saved as {output_path}')

color_index = 0
# plot for TPR scatter and linear regression
plt.figure(figsize=(15, 6))
for model_name in args.model_names:
    combined_df = read_tpr_fpr_files(base_path, subdirectories, model_name)
    x_tpr = combined_df.groupby('Gen')['TPR'].mean().sort_index().index
    y_tpr = combined_df.groupby('Gen')['TPR'].mean().sort_index().values
    x_tpr = np.asarray(x_tpr, dtype=float)
    print("x_tpr = np.asarray(x_tpr, dtype=float)")
    print(x_tpr )
    slope_tpr, intercept_tpr = np.polyfit(x_tpr, y_tpr, 1)
    regression_line_tpr = slope_tpr * x_tpr + intercept_tpr
    plt.scatter(x_tpr, y_tpr, label=f'TPR - {model_name}')
    plt.plot(x_tpr, regression_line_tpr, color=colors[color_index],label=f'TPR Regression - {model_name}')
    color_index = color_index + 1

plt.xlabel('Generation')
plt.ylabel('Average TPR Value')
plt.title('TPR Scatter and Linear Regression for Multiple Models')
plt.legend()
plt.ylim(0.9, 1)
output_path = 'images/all/combined_average_tpr_values_scatter_linear.png'
plt.savefig(output_path)
plt.close()
 
print(f'Combined TPR linear regression plot saved as {output_path}')

plt.figure(figsize=(15, 6))
color_index = 0
# plot for FPR scatter and linear regression
for model_name in args.model_names:
    combined_df = read_tpr_fpr_files(base_path, subdirectories, model_name)
    x_fpr = combined_df.groupby('Gen')['FPR'].mean().sort_index().index
    y_fpr = combined_df.groupby('Gen')['FPR'].mean().sort_index().values
    x_fpr = np.asarray(x_fpr, dtype=float)
    slope_fpr, intercept_fpr = np.polyfit(x_fpr, y_fpr, 1)
    regression_line_fpr = slope_fpr * x_fpr + intercept_fpr
    plt.scatter(x_fpr, y_fpr, label=f'FPR - {model_name}')
    plt.plot(x_fpr, regression_line_fpr, label=f'FPR Regression - {model_name}')
    color_index = color_index + 1
plt.xlabel('Generation')
plt.ylabel('Average FPR Value')
plt.ylim(0, 0.1)
plt.title('FPR Scatter and Linear Regression for Multiple Models')
plt.legend()

output_path = 'images/all/combined_average_fpr_values_scatter_linear.png'
plt.savefig(output_path)
plt.close()
 
print(f'Combined FPR linear regression plot saved as {output_path}')
