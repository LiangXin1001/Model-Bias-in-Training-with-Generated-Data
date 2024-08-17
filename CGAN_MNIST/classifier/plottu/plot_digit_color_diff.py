import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import numpy as np
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Plot color difference values across generations for a specific model')
parser.add_argument('--model_name', type=str, required=True, help='Model name for which to generate plots')
args = parser.parse_args()

# 根据模型名称定义基础路径
base_path = f'results/{args.model_name}'
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read color difference files
def read_color_diff_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'color_difference_results.csv')
        df = pd.read_csv(filepath)
        df.columns = ['Description', 'Color Difference']  # 手动设置列名
        df['Digit'] = df['Description'].str.extract(r'(\d+)').astype(int)  # 从描述中提取数字
        df['Gen'] = subdir  # 添加生成代列
        dfs.append(df[['Digit', 'Color Difference', 'Gen']])
    combined_df = pd.concat(dfs)
    return combined_df

# 读取并合并数据
combined_df = read_color_diff_files(base_path, subdirectories)

# 数据透视
pivot_df = combined_df.pivot(index='Digit', columns='Gen', values='Color Difference')

# 绘图
plt.figure(figsize=(15, 6))
pivot_df.plot(kind='bar')
plt.xlabel('Digit')
plt.ylabel('Color Difference Value')
plt.title(f'Color Difference Values for Different Digits Across Generations ({args.model_name})')
plt.legend(title='Generation')
plt.tight_layout()

# 保存图表到模型对应文件夹
output_path = f'images/{args.model_name}/digitcolor_difference_values_bar_plot.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

# 计算每代的平均颜色差异
average_cd_per_gen = combined_df.groupby('Gen')['Color Difference'].mean()

# 确保Gen列为数值类型并排序
average_cd_per_gen.index = average_cd_per_gen.index.str.extract(r'(\d+)').astype(int).squeeze()
average_cd_per_gen = average_cd_per_gen.sort_index()

# 绘制每代平均颜色差异的曲线图
plt.figure(figsize=(10, 6))
plt.ylim(0, 0.1)
plt.plot(average_cd_per_gen.index, average_cd_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Color Difference Value')
plt.title(f'Average Color Difference Values Across Generations ({args.model_name})')
plt.xticks(average_cd_per_gen.index)  # 确保x轴显示所有代数值
plt.grid(True)

# 保存曲线图到模型对应文件夹
output_path_curve = f'images/{args.model_name}/average_color_difference_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average Color Difference values curve plot saved as {output_path_curve}')

# 绘制散点图并添加线性回归曲线
x = average_cd_per_gen.index
y = average_cd_per_gen.values

# 计算线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.ylim(0, 0.1)
plt.scatter(x, y, color='blue', label='Average Color Difference')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average Color Difference Value')
plt.title(f'Average Color Difference Values Across Generations ({args.model_name})')
plt.xticks(x)  # 确保x轴显示所有代数值
plt.grid(True)
plt.legend()

# 保存散点图到模型对应文件夹
output_path_scatter = f'images/{args.model_name}/average_color_difference_values_scatter.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average Color Difference values scatter plot with linear regression saved as {output_path_scatter}')




# # 绘制散点图并添加二次回归曲线
# x = average_cd_per_gen.index
# y = average_cd_per_gen.values

# # 计算二次回归
# coefficients = np.polyfit(x, y, 2)
# polynomial = np.poly1d(coefficients)
# regression_line = polynomial(x)

# plt.figure(figsize=(10, 6))
# plt.ylim(0, 0.1)
# plt.scatter(x, y, color='blue', label='Average Color Difference')
# plt.plot(x, regression_line, color='red', label=f'Quadratic Regression\n(y={coefficients[0]:.4f}x^2 + {coefficients[1]:.4f}x + {coefficients[2]:.4f})')
# plt.xlabel('Generation')
# plt.ylabel('Average Color Difference Value')
# plt.title('Average Color Difference Values Across Generations')
# plt.xticks(x)  # 确保 x 轴显示所有代值
# plt.grid(True)
# plt.legend()

# # 保存图表
# output_path_scatter = 'average_color_difference_values_scatter_quadratic.png'
# plt.savefig(output_path_scatter)
# plt.close()

# print(f'Average Color Difference values scatter plot with quadratic regression saved as {output_path_scatter}')