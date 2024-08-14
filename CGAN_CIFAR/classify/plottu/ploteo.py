import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_name', type=str, required=True, help='The name of the model to process results for')
args = parser.parse_args()
base_path = 'metric_results'
os.makedirs(f'images/{args.model_name}', exist_ok=True) 
subdirectories = [str(i) for i in range(11)]

def read_eo_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, args.model_name, f'eo_results_{subdir}.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir # Label generation with 'gen' prefix for clarity in plots
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_eo_files(base_path, subdirectories)

# Pivot the DataFrame for easier plotting
pivot_df = combined_df.pivot_table(index='True Superclass Name', columns='Gen', values='EO', aggfunc='mean')

# Plotting bar graph
pivot_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0 , 1.0)  # Adjust the y-axis limit to enhance visualization of EO values
plt.xlabel('Superclass')
plt.ylabel('EO Value')
plt.title('EO Values for Different Superclasses Across Generations')
plt.legend(title='Generation', loc='upper left')
plt.tight_layout()

output_path = f'images/{args.model_name}/eo_values_bar_plot.png'
plt.savefig(output_path)
plt.close()
print(f'EO values bar plot saved as {output_path}')

# Calculate average EO per generation
average_eo_per_gen = combined_df.groupby('Gen')['EO'].mean()

# Ensure the 'Gen' index is sorted numerically
average_eo_per_gen.index = average_eo_per_gen.index.str.extract('(\d+)')[0].astype(int).sort_index()

# Plot the average EO values across generations
plt.figure(figsize=(10, 6))
plt.ylim(0 , 1.0)
plt.plot(average_eo_per_gen.index, average_eo_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations')
plt.xticks(average_eo_per_gen.index)  # Ensure x-axis displays all generation numbers
plt.grid(True)

output_path_curve = f'images/{args.model_name}/average_eo_values_curve.png'
plt.savefig(output_path_curve)
plt.close()
print(f'Average EO values curve plot saved as {output_path_curve}')
 

average_eo_per_gen.sort_index()

# 绘制散点图并添加线性回归直线
x = average_eo_per_gen.index
y = average_eo_per_gen.values

# 计算线性回归
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.ylim(0 , 1)
plt.scatter(x, y, color='blue', label='Average EO Value')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
plt.grid(True)
plt.legend()

# 保存图像
output_path_scatter = f'images/{args.model_name}/average_eo_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average EO values scatter plot with linear regression saved as {output_path_scatter}')