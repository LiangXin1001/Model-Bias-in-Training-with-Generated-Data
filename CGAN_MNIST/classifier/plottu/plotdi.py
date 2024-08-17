import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Plot disparate impact values across generations for a specific model')
parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model name to use for plotting')
args = parser.parse_args()

# Define the base path based on the model name
base_path = f'results/{args.model_name}'
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read disparate impact files
def read_di_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'disparate_impact_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_di_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_df = combined_df.pivot_table(index='Digit', columns='Gen', values='Disparate Impact', aggfunc='mean')

# Plotting Disparate Impact values for different digits across generations
plt.figure(figsize=(15, 6))
pivot_df.plot(kind='bar')
plt.xlabel('Digit')
plt.ylim(0.85, 1)
plt.ylabel('Disparate Impact Value')
plt.title(f'Disparate Impact Values for Different Digits Across Generations ({args.model_name})')
plt.legend(title='Generation')
plt.tight_layout()

# Save the bar plot to the model-specific directory
output_path = f'images/{args.model_name}/disparate_impact_values_bar_plot.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

# Calculate the average Disparate Impact for each generation
average_di_per_gen = combined_df.groupby('Gen')['Disparate Impact'].mean()

# Ensure the Gen column is numeric and sort by it
average_di_per_gen.index = average_di_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()
average_di_per_gen = average_di_per_gen.sort_index()

# Plot the average Disparate Impact values across generations
plt.figure(figsize=(10, 6))
plt.plot(average_di_per_gen.index, average_di_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title(f'Average Disparate Impact Values Across Generations ({args.model_name})')
plt.xticks(average_di_per_gen.index)  # Ensure x-axis shows all gen values
plt.ylim(0.85, 1)  # Set y-axis range
plt.grid(True)

# Save the curve plot to the model-specific directory
output_path_curve = f'images/{args.model_name}/average_disparate_impact_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average Disparate Impact values curve plot saved as {output_path_curve}')

# Scatter plot with linear regression
x = average_di_per_gen.index
y = average_di_per_gen.values

# Compute linear regression
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Average Disparate Impact')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title(f'Average Disparate Impact Values Across Generations ({args.model_name})')
plt.xticks(x)  # Ensure x-axis shows all gen values
plt.ylim(0.85, 1)  # Set y-axis range
plt.grid(True)
plt.legend()

# Save the scatter plot to the model-specific directory
output_path_scatter = f'images/{args.model_name}/average_disparate_impact_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average Disparate Impact values scatter plot with linear regression saved as {output_path_scatter}')
