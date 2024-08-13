import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str, nargs='+', required=True, help='The names of the models to process results for')
args = parser.parse_args()
# Define the base path

# Function to read all images result files and extract generation from filenames
def read_all_images_result_files(base_path, subdirectories,model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f'test_results_{subdir}.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir  # Add generation column extracted from filename
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

def process_model_data(model_name):
    base_path = 'results'
    subdirectories = [f'{i}' for i in range(11)]
    
    # Read and combine data
    combined_df = read_all_images_result_files(base_path, subdirectories)

    # Calculate accuracy for each row
    combined_df['Correct'] = combined_df['True Superclass'] == combined_df['Predicted Superclass']

    # Calculate the average accuracy for each gen
    average_accuracy_per_gen = combined_df.groupby('Gen')['Correct'].mean()

    # Ensure the Gen column is numeric and sort by it
    average_accuracy_per_gen.index = average_accuracy_per_gen.index.astype(int)
    average_accuracy_per_gen = average_accuracy_per_gen.sort_index()
    return average_accuracy_per_gen 




# Plot the average accuracy values across generations
plt.figure(figsize=(10, 6)) 
for model_name in args.model_names:
    average_accuracy_per_gen = process_model_data(model_name)
    plt.plot(average_accuracy_per_gen.index, average_accuracy_per_gen.values, marker='o', label=model_name)

plt.ylim(0 , 1.0)
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations  for Multiple Models')
plt.xticks(average_accuracy_per_gen.index)  # Ensure x-axis shows all gen values
plt.grid(True)

# Save the plot
 
os.makedirs(f'images/all', exist_ok=True)
output_path_curve = f'images/all/average_accuracy_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average accuracy curve plot saved as {output_path_curve}')

# Plotting scatter and adding linear regression line
plt.figure(figsize=(10, 6)) 

colors = ['blue', 'green', 'red', 'purple', 'orange']  # Define a list of colors for different models
color_index = 0

for model_name in args.model_names:
    average_accuracy_per_gen = process_model_data(model_name)
    x = average_accuracy_per_gen.index
    y = average_accuracy_per_gen.values
    # Calculate linear regression
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Average Accuracy')
    plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
 
plt.ylim(0 , 1.0)
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations')
plt.xticks(x)  # Ensure x-axis shows all generation values
plt.grid(True)
plt.legend()

# Save the image
output_path_scatter = f'images/all/average_accuracy_scatter_linear.png'
if not os.path.exists('images/all'):
    os.makedirs('images/all')
plt.savefig(output_path_scatter)
plt.close()

print(f'Average accuracy scatter plot with linear regression saved as {output_path_scatter}')
