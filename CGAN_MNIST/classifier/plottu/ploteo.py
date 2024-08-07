import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# Define the base path
base_path = 'results'

# Define subdirectories
subdirectories = ['gen{}'.format(i) for i in range(11)]

 
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

# Plotting
pivot_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0.98, 1.0)
plt.xlabel('Digit')
plt.ylabel('EO Value')
plt.title('EO Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

output_path = 'plottu/eo_values_bar_plot.png'
plt.savefig(output_path)
plt.close()


# calculate average EO

# 计算每个gen中十个数字的EO平均值
average_eo_per_gen = combined_df.groupby('Gen')['EO'].mean()

# 将Gen列转换为数字类型以确保正确排序
average_eo_per_gen.index = average_eo_per_gen.index.str.extract('(\d+)').astype(int).squeeze()

# 按Gen排序
average_eo_per_gen = average_eo_per_gen.sort_index()

# 绘制平均值随gen变化的曲线图
plt.figure(figsize=(10, 6))
plt.ylim(0.9, 1)
plt.plot(average_eo_per_gen.index, average_eo_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations')
plt.xticks(average_eo_per_gen.index)  # 确保x轴显示所有gen值
plt.grid(True)

# 保存图像
output_path_curve = 'plottu/average_eo_values_curve.png'
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
plt.ylim(0.9, 1)
plt.scatter(x, y, color='blue', label='Average EO Value')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
plt.grid(True)
plt.legend()

# 保存图像
output_path_scatter = 'plottu/average_eo_values_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average EO values scatter plot with linear regression saved as {output_path_scatter}')