import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_name', type=str, required=True, help='The name of the model to process results for')
args = parser.parse_args()
base_path = 'metric_results'
os.makedirs(base_path, exist_ok=True) 
subdirectories = [str(i) for i in range(11)]
 
def read_color_diff_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, args.model_name, f'subclass_difference_results_{subdir}.csv')
        df = pd.read_csv(filepath)
        # df.columns = ['Index', 'Superclass', 'Subclass with Max Accuracy', 'Subclass with Min Accuracy', 'Max Difference'] # 手动设置列名
        # df['True Superclass Name'] = df['Description'].str.extract(r'(\d+)').astype(int)  # 从描述中提取数字
        df['Gen'] = subdir  # 添加生成代列
        dfs.append(df[['True Superclass Name', 'Max Difference', 'Gen']])
    combined_df = pd.concat(dfs)
    return combined_df

# 读取并合并数据
combined_df = read_color_diff_files(base_path, subdirectories)

# 数据透视
pivot_df = combined_df.pivot(index='True Superclass Name', columns='Gen', values='Max Difference')

# 绘图
plt.figure(figsize=(15, 6))
pivot_df.plot(kind='bar')
plt.xlabel('True Superclass Name')
plt.ylabel('Subclass Difference Value')
plt.title('Subclass Difference Values for Different True Superclass Names Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

# 保存图表
output_path = f'images/{args.model_name}/Superclass_subclass_difference_values_bar_plot.png'
plt.savefig(output_path)
plt.close()
 


# Calculate the average Subclass Difference for each gen
average_cd_per_gen = combined_df.groupby('Gen')['Max Difference'].mean()

# Ensure the Gen column is numeric and sort by it
average_cd_per_gen.index = average_cd_per_gen.index.str.extract('(\d+)').astype(int).squeeze()
average_cd_per_gen = average_cd_per_gen.sort_index()

# Plot the average Subclass Difference values across generations
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)
plt.plot(average_cd_per_gen.index, average_cd_per_gen.values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title('Average Subclass Difference Values Across Generations')
plt.xticks(average_cd_per_gen.index)  # Ensure x-axis shows all gen values
plt.grid(True)

# Save the plot
output_path_curve = f'images/{args.model_name}/average_subclass_difference_values_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average Subclass Difference values curve plot saved as {output_path_curve}')




# 绘制散点图并添加线性回归曲线
x = average_cd_per_gen.index
y = average_cd_per_gen.values

# 计算线性回归
slope, intercept, r_value, p_value, std_err = linregress(x, y)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.ylim(0,  1)
plt.scatter(x, y, color='blue', label='Average Subclass Difference')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title('Average Subclass Difference Values Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
plt.grid(True)
plt.legend()

# 保存图表
output_path_scatter = f'images/{args.model_name}/average_subclass_difference_values_scatter.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average Subclass Difference values scatter plot with linear regression saved as {output_path_scatter}')




# # 绘制散点图并添加二次回归曲线
# x = average_cd_per_gen.index
# y = average_cd_per_gen.values

# # 计算二次回归
# coefficients = np.polyfit(x, y, 2)
# polynomial = np.poly1d(coefficients)
# regression_line = polynomial(x)

# plt.figure(figsize=(10, 6))
# plt.ylim(0, 0.1)
# plt.scatter(x, y, color='blue', label='Average Subclass Difference')
# plt.plot(x, regression_line, color='red', label=f'Quadratic Regression\n(y={coefficients[0]:.4f}x^2 + {coefficients[1]:.4f}x + {coefficients[2]:.4f})')
# plt.xlabel('Generation')
# plt.ylabel('Average Subclass Difference Value')
# plt.title('Average Subclass Difference Values Across Generations')
# plt.xticks(x)  # 确保 x 轴显示所有代值
# plt.grid(True)
# plt.legend()

# # 保存图表
# output_path_scatter = 'average_subclass_difference_values_scatter_quadratic.png'
# plt.savefig(output_path_scatter)
# plt.close()

# print(f'Average Subclass Difference values scatter plot with quadratic regression saved as {output_path_scatter}')