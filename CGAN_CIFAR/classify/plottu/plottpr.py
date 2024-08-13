import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_name', type=str, required=True, help='The name of the model to process results for')
args = parser.parse_args()

# Define the base path
base_path = 'metric_results'
subdirectories = [str(i) for i in range(11)]

# Function to read TPR and FPR files
def read_tpr_fpr_files(base_path, subdirectories, model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f'tpr_fpr_results_{subdir}.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_tpr_fpr_files(base_path, subdirectories, args.model_name)

# Pivot the DataFrame for plotting
pivot_tpr_df = combined_df.pivot_table(index='True Superclass Name', columns='Gen', values='TPR', aggfunc='mean')
pivot_fpr_df = combined_df.pivot_table(index='True Superclass Name', columns='Gen', values='FPR', aggfunc='mean')

# Plotting TPR
plt.figure(figsize=(15, 6))
pivot_tpr_df.plot(kind='bar')
plt.ylim(0, 1)  # Adjust the y-axis limit if necessary
plt.xlabel('True Superclass')
plt.ylabel('TPR Value')
plt.title('TPR Values for Different Superclasses Across Generations')
plt.legend(title='Generation')
plt.tight_layout()
tpr_output_path = 'images/tpr_values_bar_plot.png'
plt.savefig(tpr_output_path)
plt.close()

# Plotting FPR
plt.figure(figsize=(15, 6))
pivot_fpr_df.plot(kind='bar')
plt.ylim(0, 1)  # Adjust the y-axis limit if necessary
plt.xlabel('True Superclass')
plt.ylabel('FPR Value')
plt.title('FPR Values for Different Superclasses Across Generations')
plt.legend(title='Generation')
plt.tight_layout()
fpr_output_path = 'images/fpr_values_bar_plot.png'
plt.savefig(fpr_output_path)
plt.close()

# Calculate average TPR and FPR per generation
average_tpr_per_gen = combined_df.groupby('Gen')['TPR'].mean()
average_fpr_per_gen = combined_df.groupby('Gen')['FPR'].mean()

# Ensure the Gen column is numeric for sorting
average_tpr_per_gen.index = average_tpr_per_gen.index.astype(int)
average_fpr_per_gen.index = average_fpr_per_gen.index.astype(int)

# Sort by generation
average_tpr_per_gen = average_tpr_per_gen.sort_index()
average_fpr_per_gen = average_fpr_per_gen.sort_index()

# Plot average TPR values across generations
plt.figure(figsize=(10, 6))
plt.plot(average_tpr_per_gen.index, average_tpr_per_gen.values, marker='o', label='TPR')
plt.xlabel('Generation')
plt.ylabel('Average TPR Value')
plt.title('Average TPR Values Across Generations')
plt.xticks(average_tpr_per_gen.index)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
tpr_output_path_curve = 'images/average_tpr_values_curve.png'
plt.savefig(tpr_output_path_curve)
plt.close()

# Plot average FPR values across generations
plt.figure(figsize=(10, 6))
plt.plot(average_fpr_per_gen.index, average_fpr_per_gen.values, marker='o', label='FPR')
plt.xlabel('Generation')
plt.ylabel('Average FPR Value')
plt.title('Average FPR Values Across Generations')
plt.xticks(average_fpr_per_gen.index)
plt.ylim(0, 0.1)
plt.grid(True)
plt.legend()
fpr_output_path_curve = 'images/average_fpr_values_curve.png'
plt.savefig(fpr_output_path_curve)
plt.close()

print(f'Average TPR values curve plot saved as {tpr_output_path_curve}')
print(f'Average FPR values curve plot saved as {fpr_output_path_curve}')
 


# 绘制TPR散点图并添加线性回归直线
x_tpr = average_tpr_per_gen.index
y_tpr = average_tpr_per_gen.values

# 计算线性回归
slope_tpr, intercept_tpr = np.polyfit(x_tpr, y_tpr, 1)
regression_line_tpr = slope_tpr * x_tpr + intercept_tpr

plt.figure(figsize=(10, 6))
plt.scatter(x_tpr, y_tpr, color='blue', label='Average TPR Value')
plt.plot(x_tpr, regression_line_tpr, color='red', label=f'Linear Regression\n(y={slope_tpr:.4f}x + {intercept_tpr:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average TPR Value')
plt.title('Average TPR Values Across Generations')
plt.xticks(x_tpr)  # 确保x轴显示所有gen值
plt.ylim(0 , 1)  # 设置y轴范围
plt.grid(True)
plt.legend()

# 保存TPR散点图
tpr_output_path_scatter = 'images/average_tpr_values_scatter_linear.png'
plt.savefig(tpr_output_path_scatter)
plt.close()

# 绘制FPR散点图并添加线性回归直线
x_fpr = average_fpr_per_gen.index
y_fpr = average_fpr_per_gen.values

# 计算线性回归
slope_fpr, intercept_fpr = np.polyfit(x_fpr, y_fpr, 1)
regression_line_fpr = slope_fpr * x_fpr + intercept_fpr

plt.figure(figsize=(10, 6))
plt.scatter(x_fpr, y_fpr, color='blue', label='Average FPR Value')
plt.plot(x_fpr, regression_line_fpr, color='red', label=f'Linear Regression\n(y={slope_fpr:.4f}x + {intercept_fpr:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average FPR Value')
plt.title('Average FPR Values Across Generations')
plt.xticks(x_fpr)  # 确保x轴显示所有gen值
plt.ylim(0, 0.1)  # 设置y轴范围
plt.grid(True)
plt.legend()

# 保存FPR散点图
fpr_output_path_scatter = 'images/average_fpr_values_scatter_linear.png'
plt.savefig(fpr_output_path_scatter)
plt.close()

print(f'Average TPR values scatter plot with linear regression saved as {tpr_output_path_scatter}')
print(f'Average FPR values scatter plot with linear regression saved as {fpr_output_path_scatter}')