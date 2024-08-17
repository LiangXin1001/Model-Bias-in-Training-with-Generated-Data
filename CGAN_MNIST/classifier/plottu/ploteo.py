import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Plot EO values for different generations')
parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
args = parser.parse_args()

# Define the base path based on the model name
base_path = f'results/{args.model_name}'

# Define subdirectories (generation folders)
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read EO files
def read_eo_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'eo_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_eo_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_df = combined_df.pivot(index='Digit', columns='Gen', values='EO')

# Plotting EO values for different digits across generations
pivot_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0.98, 1.0)
plt.xlabel('Digit')
plt.ylabel('EO Value')
plt.title(f'EO Values for Different Digits Across Generations ({args.model_name})')
plt.legend(title='Generation')
plt.tight_layout()

# Save the bar plot to the model-specific directory
output_path = f'images/{args.model_name}/eo_values_bar_plot.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

# Calculate average EO per generation
average_eo_per_gen = combined_df.groupby('Gen')['EO'].mean()

# Convert Gen column to numeric for proper sorting
average_eo_per_gen.index = average_eo_per_gen.index.str.extract('(\d+)').astype(int).squeeze()

# Sort by Gen
average_eo_per_gen = average_eo_per_gen.sort_index()

# Plot the average EO values across generations
plt.figure(figsize=(10, 6))
plt.ylim(0.9, 1)
plt.plot(average_eo_per_gen.index, average_eo_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title(f'Average EO Values Across Generations ({args.model_name})')
plt.xticks(average_eo_per_gen.index)  # Ensure x-axis shows all gen values
plt.grid(True)

# Save the curve plot to the model-specific directory
output_path_curve = f'images/{args.model_name}/average_eo_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average EO values curve plot saved as {output_path_curve}')

# Scatter plot with linear regression
x = average_eo_per_gen.index
y = average_eo_per_gen.values

# Compute linear regression
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.ylim(0.9, 1)
plt.scatter(x, y, color='blue', label='Average EO Value')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title(f'Average EO Values Across Generations ({args.model_name})')
plt.xticks(x)  # Ensure x-axis shows all gen values
plt.grid(True)
plt.legend()

# Save the scatter plot to the model-specific directory
output_path_scatter = f'images/{args.model_name}/average_eo_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average EO values scatter plot with linear regression saved as {output_path_scatter}')
