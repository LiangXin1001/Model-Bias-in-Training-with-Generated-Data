 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str, nargs='+', required=True, help='The names of the models to process results for')
args = parser.parse_args()
base_path = 'results'
os.makedirs(base_path, exist_ok=True) 
subdirectories = [str(i) for i in range(11)]

# Function to read disparate impact files
def read_di_files(base_path, subdirectories,model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f"gen{subdir}",  f'disparate_impact_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

colors = ['blue', 'green', 'red', 'purple', 'orange']   
color_index = 0
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    # Read and combine data
    combined_df = read_di_files(base_path, subdirectories,model_name)
    # Pivot the DataFrame for plotting
    pivot_df = combined_df.pivot_table(index='Digit', columns='Gen', values='Disparate Impact', aggfunc='mean')
    # 计算每个gen中十个数字的Disparate Impact平均值
    average_di_per_gen = combined_df.groupby('Gen')['Disparate Impact'].mean()
    # 将Gen列转换为数字类型以确保正确排序
    average_di_per_gen.index = average_di_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()
    # 按Gen排序
    average_di_per_gen = average_di_per_gen.sort_index()
    plt.plot(average_di_per_gen.index, average_di_per_gen.values , marker='o', label=model_name)
    # color_index = color_index + 1

plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Values Across Generations')
plt.xticks(average_di_per_gen.index)  # 确保x轴显示所有gen值
 
plt.grid(True)

# 保存图像
output_path_curve = f'images/all/average_disparate_impact_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average Disparate Impact values curve plot saved as {output_path_curve}')


color_index = 0
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    # Read and combine data
    combined_df = read_di_files(base_path, subdirectories,model_name)
    # Pivot the DataFrame for plotting
    pivot_df = combined_df.pivot_table(index='Digit', columns='Gen', values='Disparate Impact', aggfunc='mean')
    # 计算每个gen中十个数字的Disparate Impact平均值
    average_di_per_gen = combined_df.groupby('Gen')['Disparate Impact'].mean()
    # 将Gen列转换为数字类型以确保正确排序
    average_di_per_gen.index = average_di_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()
    # 按Gen排序
    average_di_per_gen = average_di_per_gen.sort_index()
    # 绘制散点图并添加线性回归直线
    x = average_di_per_gen.index
    y = average_di_per_gen.values
    # 计算线性回归
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept
    
    plt.scatter(x, y, color='blue', label=f'{model_name}Average Disparate Impact')
    plt.plot(x, regression_line,  label=f'{model_name} Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
    # color_index = color_index + 1


plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Values Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
 
plt.grid(True)
plt.legend()

output_dir = 'images/all'
os.makedirs(output_dir, exist_ok=True)
output_path_scatter = f'images/all/average_disparate_impact_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average Disparate Impact values scatter plot with linear regression saved as {output_path_scatter}')