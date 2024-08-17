 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_name', type=str, required=True, help='The name of the model to process results for')
args = parser.parse_args()
base_path = 'metric_results'
os.makedirs(base_path, exist_ok=True) 
subdirectories = [str(i) for i in range(11)]
os.makedirs(f'images/{args.model_name}', exist_ok=True) 
# Function to read disparate impact files
def read_di_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, args.model_name, f'disparate_impact_results_{subdir}.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_di_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_df = combined_df.pivot_table(index='True Superclass Name', columns='Gen', values='Disparate Impact', aggfunc='mean')

# Plotting
pivot_df.plot(kind='bar', figsize=(15, 6))
 
plt.xlabel('True Superclass Name')
plt.ylim(0 , 1)
plt.ylabel('Disparate Impact Value')
plt.title('Disparate Impact Values for Different True Superclass Names Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

output_path = f'images/{args.model_name}/disparate_impact_values_bar_plot.png'
plt.savefig(output_path)
plt.close()
 
# 计算每个gen中十个数字的Disparate Impact平均值
average_di_per_gen = combined_df.groupby('Gen')['Disparate Impact'].mean()

# 将Gen列转换为数字类型以确保正确排序
average_di_per_gen.index = average_di_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()

# 按Gen排序
average_di_per_gen = average_di_per_gen.sort_index()

# 绘制平均值随gen变化的曲线图
plt.figure(figsize=(10, 6))
plt.plot(average_di_per_gen.index, average_di_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Values Across Generations')
plt.xticks(average_di_per_gen.index)  # 确保x轴显示所有gen值
plt.ylim(0 , 1)  # 设置y轴范围
plt.grid(True)

# 保存图像
output_path_curve = f'images/{args.model_name}/average_disparate_impact_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average Disparate Impact values curve plot saved as {output_path_curve}')





# 绘制散点图并添加线性回归直线
x = average_di_per_gen.index
y = average_di_per_gen.values

# 计算线性回归
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Average Disparate Impact')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Values Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
plt.ylim(0 , 1)  # 设置 y 轴范围
plt.grid(True)
plt.legend()

# 保存图表
output_path_scatter = f'images/{args.model_name}/average_disparate_impact_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average Disparate Impact values scatter plot with linear regression saved as {output_path_scatter}')