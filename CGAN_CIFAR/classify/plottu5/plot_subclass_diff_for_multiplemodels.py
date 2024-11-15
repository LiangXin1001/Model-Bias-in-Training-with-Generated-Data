import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import numpy as np
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str, nargs='+', required=True, help='The names of the models to process results for')
args = parser.parse_args()
base_path = 'results'
os.makedirs(base_path, exist_ok=True) 
subdirectories = [str(i) for i in range(11)]
output_dir = "images/all"
os.makedirs(output_dir, exist_ok=True)

def read_color_diff_files(base_path, subdirectories,model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f"gen{subdir}", 'color_difference_results.csv')
        df = pd.read_csv(filepath)
        df.columns = ['Description', 'Color Difference']  # 手动设置列名
        df['Digit'] = df['Description'].str.extract(r'(\d+)').astype(int)  # 从描述中提取数字
        df['Gen'] = subdir  # 添加生成代列
        dfs.append(df[['Digit', 'Color Difference', 'Gen']])
    combined_df = pd.concat(dfs)
    return combined_df

colors = ['blue', 'green', 'red', 'purple', 'orange']   
color_index = 0

# 绘制每个模型的数据
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    combined_df = read_color_diff_files(base_path, subdirectories, model_name)

    # Calculate the average Subclass Difference for each gen
    average_cd_per_gen = combined_df.groupby('Gen')['Color Difference'].mean()
    average_cd_per_gen.index = average_cd_per_gen.index.astype(int)
    average_cd_per_gen = average_cd_per_gen.sort_index()
    # Plot the average Subclass Difference values across generations
    plt.plot(average_cd_per_gen.index, average_cd_per_gen.values, color=colors[color_index],marker='o', label=f'{model_name}')
    color_index = color_index +1

 
 
plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title(f'Average Subclass Difference Values Across Generations for Multiple Models ')
plt.xticks(average_cd_per_gen.index)
plt.grid(True)
output_path_curve = os.path.join(output_dir, f'average_subclass_difference_curve.png')
plt.savefig(output_path_curve)
plt.close()
print(f'Average Subclass Difference values curve plot for  all model saved  ')

color_index = 0
plt.figure(figsize=(10, 6))
for model_name in args.model_names:
    combined_df = read_color_diff_files(base_path, subdirectories, model_name)

    # Calculate the average Subclass Difference for each gen
    average_cd_per_gen = combined_df.groupby('Gen')['Color Difference'].mean()
    average_cd_per_gen.index = average_cd_per_gen.index.astype(int)
    average_cd_per_gen = average_cd_per_gen.sort_index()
 
    # Scatter plot with linear regression
    x = average_cd_per_gen.index
    y = average_cd_per_gen.values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept
    plt.scatter(x, y, color='blue', label='Average Subclass Difference')
    plt.plot(x, regression_line, color=colors[color_index], label=f'{model_name} Regression (y={slope:.4f}x + {intercept:.4f})' )
    color_index = color_index +1

 

plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title(f'Linear Regression of Subclass Differences for Multiple Models')
plt.xticks(x)
plt.grid(True)
plt.legend()
output_path_scatter = os.path.join(output_dir, f'average_subclass_difference_scatter.png')
plt.savefig(output_path_scatter)
plt.close()
print(f'Average Subclass Difference scatter plot with linear regression for all models saved as {output_dir}')
   

 