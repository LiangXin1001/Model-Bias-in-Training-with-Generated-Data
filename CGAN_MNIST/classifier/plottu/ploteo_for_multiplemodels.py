import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

 
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str, nargs='+', required=True, help='The names of the models to process results for')
args = parser.parse_args()
base_path = 'results'
os.makedirs(base_path, exist_ok=True)
subdirectories = [str(i) for i in range(11)]

def read_eo_files(base_path, subdirectories, model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name,f"gen{subdir}", f'eo_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir  # Label generation with 'gen' prefix for clarity in plots
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df
 

images_dir = "images/all"
os.makedirs(images_dir, exist_ok=True)
 
colors = ['blue', 'green', 'red', 'purple', 'orange']   
color_index = 0

# Average EO values across generations
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    combined_df = read_eo_files(base_path, subdirectories, model_name)
    average_eo_per_gen = combined_df.groupby('Gen')['EO'].mean()
    plt.plot(average_eo_per_gen.index, average_eo_per_gen.values,color=colors[color_index], marker='o', label=model_name)
    color_index = color_index + 1
plt.ylim(0.95, 1.0)
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations for Multiple Models')
plt.xticks(average_eo_per_gen.index)
plt.grid(True)
plt.legend()
output_path_curve = f'{images_dir}/average_eo_values_curve.png'
plt.savefig(output_path_curve)
plt.close()
color_index =0
# Scatter plot with linear regression
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    combined_df = read_eo_files(base_path, subdirectories, model_name)
    average_eo_per_gen = combined_df.groupby('Gen')['EO'].mean()
    x = average_eo_per_gen.index
    y = average_eo_per_gen.values
    x = np.asarray(x, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept
    plt.scatter(x, y,color=colors[color_index], label=f'{model_name} Average EO Value')
    plt.plot(x, regression_line, color=colors[color_index],label=f'{model_name} Linear Regression (y={slope:.4f}x + {intercept:.4f})')
    color_index = color_index + 1

plt.ylim(0.95, 1)
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations for Multiple Models')
plt.xticks(x)
plt.grid(True)
plt.legend()
output_path_scatter = f'{images_dir}/average_eo_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print("average_eo_values_scatter_linear has been saved")