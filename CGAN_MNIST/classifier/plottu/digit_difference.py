import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Plot digit differences across generations')
parser.add_argument('--model_name', type=str, required=True, help='Model name for which to generate plots')
args = parser.parse_args()

# Define the base path based on the model name
base_path = f'results/{args.model_name}'
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read digit difference files
def read_digit_diff_files(base_path, subdirectories,model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path,    subdir, 'digit_difference_results.csv')
        df = pd.read_csv(filepath, header=None, names=['Description', 'Value'])
        df['Gen'] = subdir  # Add generation column
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df


def digit_difference(base_path, model_name):
    # Read and combine data
    combined_df = read_digit_diff_files(base_path, subdirectories,model_name)

    # Extract the color and digit difference from the Description
    combined_df = combined_df.dropna(subset=['Description'])
    combined_df['Color Digit Difference'] = combined_df['Description'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])

    # Pivot the DataFrame for calculating means
    pivot_df = combined_df.pivot_table(index='Gen', columns='Color Digit Difference', values='Value', aggfunc='mean')

    # Ensure the Gen column is numeric and sort by it
    pivot_df.index = pivot_df.index.str.extract('(\d+)').astype(int).squeeze()
    pivot_df = pivot_df.sort_index()

    # Plotting the average digit differences across generations
    plt.figure(figsize=(10, 6))
    for col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], marker='o', label=col)
    plt.ylim(0, 0.2)
    plt.xlabel('Generation')
    plt.ylabel('Average Digit Difference')
    plt.title(f'Average Digit Difference Across Generations ({model_name})')
    plt.legend(title='Color Digit Difference')
    plt.grid(True)

    # Save the plot to the model-specific directory
    output_path_curve = f'images/{model_name}/average_digit_difference_curve.png'
    os.makedirs(os.path.dirname(output_path_curve), exist_ok=True)
    plt.savefig(output_path_curve)
    plt.close()

    print(f'Average digit difference curve plot saved as {output_path_curve}')


read_digit_diff_files(base_path,subdirectories,args.model_name)
digit_difference(base_path, args.model_name)