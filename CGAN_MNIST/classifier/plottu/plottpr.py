import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

 
# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
args = parser.parse_args()
 
# Function to read TPR and FPR files with dynamic paths
def read_tpr_fpr_files(result_base_path, model_name, generations):
    dfs = []
    for gen in generations:
        filepath = os.path.join(result_base_path, model_name, f'gen{gen}', 'tpr_fpr_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = f'gen{gen}'
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Function to plot TPR and FPR for multiple generations
def plot_tpr_fpr(base_path, model_name, generations):
    combined_df = read_tpr_fpr_files(base_path, model_name, generations)
    
    # Pivot the DataFrame for plotting
    pivot_tpr_df = combined_df.pivot_table(index='Digit', columns='Gen', values='TPR', aggfunc='mean')
    pivot_fpr_df = combined_df.pivot_table(index='Digit', columns='Gen', values='FPR', aggfunc='mean')

    # Plotting TPR
    pivot_tpr_df.plot(kind='bar', figsize=(15, 6))
    plt.ylim(0.4, 1)  # Adjust the y-axis limit if necessary
    plt.xlabel('Digit')
    plt.ylabel('TPR Value')
    plt.title('TPR Values for Different Digits Across Generations')
    plt.legend(title='Generation')
    plt.tight_layout()
    tpr_output_path = f'images/{model_name}/tpr_values_bar_plot.png'
    os.makedirs(os.path.dirname(tpr_output_path), exist_ok=True)
    plt.savefig(tpr_output_path)
    plt.close()

    # Plotting FPR
    pivot_fpr_df.plot(kind='bar', figsize=(15, 6))
    plt.ylim(0, 1)  # Adjust the y-axis limit if necessary
    plt.xlabel('Digit')
    plt.ylabel('FPR Value')
    plt.title('FPR Values for Different Digits Across Generations')
    plt.legend(title='Generation')
    plt.tight_layout()
    fpr_output_path = f'images/{model_name}/fpr_values_bar_plot.png'
    plt.savefig(fpr_output_path)
    plt.close()

    # Calculate average TPR and FPR per generation
    average_tpr_per_gen = combined_df.groupby('Gen')['TPR'].mean()
    average_fpr_per_gen = combined_df.groupby('Gen')['FPR'].mean()

    # Convert Gen column to numeric for proper sorting
    average_tpr_per_gen.index = average_tpr_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()
    average_fpr_per_gen.index = average_fpr_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()

    # Sort by Gen
    average_tpr_per_gen = average_tpr_per_gen.sort_index()
    average_fpr_per_gen = average_fpr_per_gen.sort_index()

    # Plot TPR average values across generations
    plt.figure(figsize=(10, 6))
    plt.plot(average_tpr_per_gen.index, average_tpr_per_gen.values, marker='o', label='TPR')
    plt.xlabel('Generation')
    plt.ylabel('Average TPR Value')
    plt.title('Average TPR Values Across Generations')
    plt.xticks(average_tpr_per_gen.index)
    plt.ylim(0.9, 1)
    plt.grid(True)
    plt.legend()
    tpr_output_path_curve = f'images/{model_name}/average_tpr_values_curve.png'
    plt.savefig(tpr_output_path_curve)
    plt.close()

    # Plot FPR average values across generations
    plt.figure(figsize=(10, 6))
    plt.plot(average_fpr_per_gen.index, average_fpr_per_gen.values, marker='o', label='FPR')
    plt.xlabel('Generation')
    plt.ylabel('Average FPR Value')
    plt.title('Average FPR Values Across Generations')
    plt.xticks(average_fpr_per_gen.index)
    plt.ylim(0, 0.1)
    plt.grid(True)
    plt.legend()
    fpr_output_path_curve = f'images/{model_name}/average_fpr_values_curve.png'
    plt.savefig(fpr_output_path_curve)
    plt.close()

    print(f'Average TPR values curve plot saved as {tpr_output_path_curve}')
    print(f'Average FPR values curve plot saved as {fpr_output_path_curve}')

    # Plot TPR scatter plot with linear regression
    x_tpr = average_tpr_per_gen.index
    y_tpr = average_tpr_per_gen.values
    slope_tpr, intercept_tpr = np.polyfit(x_tpr, y_tpr, 1)
    regression_line_tpr = slope_tpr * x_tpr + intercept_tpr

    plt.figure(figsize=(10, 6))
    plt.scatter(x_tpr, y_tpr, color='blue', label='Average TPR Value')
    plt.plot(x_tpr, regression_line_tpr, color='red', label=f'Linear Regression\n(y={slope_tpr:.4f}x + {intercept_tpr:.4f})')
    plt.xlabel('Generation')
    plt.ylabel('Average TPR Value')
    plt.title('Average TPR Values Across Generations')
    plt.xticks(x_tpr)
    plt.ylim(0.9, 1)
    plt.grid(True)
    plt.legend()
    tpr_output_path_scatter = f'images/{model_name}/average_tpr_values_scatter_linear.png'
    plt.savefig(tpr_output_path_scatter)
    plt.close()

    # Plot FPR scatter plot with linear regression
    x_fpr = average_fpr_per_gen.index
    y_fpr = average_fpr_per_gen.values
    slope_fpr, intercept_fpr = np.polyfit(x_fpr, y_fpr, 1)
    regression_line_fpr = slope_fpr * x_fpr + intercept_fpr

    plt.figure(figsize=(10, 6))
    plt.scatter(x_fpr, y_fpr, color='blue', label='Average FPR Value')
    plt.plot(x_fpr, regression_line_fpr, color='red', label=f'Linear Regression\n(y={slope_fpr:.4f}x + {intercept_fpr:.4f})')
    plt.xlabel('Generation')
    plt.ylabel('Average FPR Value')
    plt.title('Average FPR Values Across Generations')
    plt.xticks(x_fpr)
    plt.ylim(0, 0.1)
    plt.grid(True)
    plt.legend()
    fpr_output_path_scatter = f'images/{model_name}/average_fpr_values_scatter_linear.png'
    plt.savefig(fpr_output_path_scatter)
    plt.close()

    print(f'Average TPR values scatter plot with linear regression saved as {tpr_output_path_scatter}')
    print(f'Average FPR values scatter plot with linear regression saved as {fpr_output_path_scatter}')

# Example usage
base_path = 'results'
 
generations = range(11)  # Example range of generations

plot_tpr_fpr(base_path, args.model_name, generations)
