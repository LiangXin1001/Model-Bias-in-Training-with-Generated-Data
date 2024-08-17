import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

 
# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
args = parser.parse_args()
 
 
# Function to read all images result files with dynamic paths
def read_all_images_result_files(base_path, model_name, generations):
    dfs = []
    for gen in generations:
        result_path = f"{base_path}/{model_name}/gen{gen}/all_images_results.csv"
        df = pd.read_csv(result_path)
        df['Gen'] = f"gen{gen}"  # Add generation column
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Function to plot average accuracy across generations
def plot_accuracy(base_path, model_name, generations):
    # Read and combine data
    combined_df = read_all_images_result_files(base_path, model_name, generations)
    
    # Calculate accuracy for each row
    combined_df['Correct'] = combined_df['True Label'] == combined_df['Predicted Label']
    
    # Calculate the average accuracy for each gen
    average_accuracy_per_gen = combined_df.groupby('Gen')['Correct'].mean()
    
    # Ensure the Gen column is numeric and sort by it
    average_accuracy_per_gen.index = average_accuracy_per_gen.index.str.extract('(\d+)').astype(int).squeeze()
    average_accuracy_per_gen = average_accuracy_per_gen.sort_index()
    
    # Plot the average accuracy values across generations
    plt.figure(figsize=(10, 6))
    plt.plot(average_accuracy_per_gen.index, average_accuracy_per_gen.values, marker='o')
    plt.ylim(0.8, 1.0)
    plt.xlabel('Generation')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy Across Generations')
    plt.xticks(average_accuracy_per_gen.index)  # Ensure x-axis shows all gen values
    plt.grid(True)

    # Save the plot
    output_path_curve = f'images/{model_name}/average_accuracy_curve.png'
    os.makedirs(os.path.dirname(output_path_curve), exist_ok=True)
    plt.savefig(output_path_curve)
    plt.close()
    print(f'Average accuracy curve plot saved as {output_path_curve}')
    
    # Plot scatter with linear regression
    x = average_accuracy_per_gen.index
    y = average_accuracy_per_gen.values

    # Calculate linear regression
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Average Accuracy')
    plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
    plt.ylim(0.8, 1.0)
    plt.xlabel('Generation')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy Across Generations')
    plt.xticks(x)  # Ensure x-axis shows all gen values
    plt.grid(True)
    plt.legend()

    # Save the scatter plot
    output_path_scatter = f'images/{model_name}/average_accuracy_scatter_linear.png'
    plt.savefig(output_path_scatter)
    plt.close()
    print(f'Average accuracy scatter plot with linear regression saved as {output_path_scatter}')
 
 
base_path = 'results'
 
generations = range(11)  # Example range of generations

plot_accuracy(base_path, args.model_name, generations)
