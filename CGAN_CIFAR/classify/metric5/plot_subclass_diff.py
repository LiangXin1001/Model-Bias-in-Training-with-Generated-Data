import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import numpy as np
import argparse

  
# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--output_path', type=str, required=True, help='Path to save resulting plots')
args = parser.parse_args()

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']

# 设置基本路径
base_dir = args.result_dir
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)
subdirectories = [str(i) for i in range(11)]

# 读取子类差异值文件的函数
def read_color_diff_files(base_path, subdirectories, model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f'subclass_difference_results_{subdir}.csv')
        if os.path.exists(filepath):
           
            print(f"Reading file: {filepath}")  # 打印正在读取的文件名
            df = pd.read_csv(filepath)
            print(df.head())  # 打印文件前几行以检查数据内容
            df['Gen'] = int(subdir)  # 添加生成代次信息
            dfs.append(df[['Run ID', 'True Superclass Name', 'Max Difference', 'Gen']])
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        raise ValueError(f" none in df")
         

colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', 'D', '^', 'v']

# 创建一个画布，用于绘制所有模型的曲线
plt.figure(figsize=(12, 8))

# 遍历每个模型，读取和绘制数据
for model_idx, model_name in enumerate(model_names):
    path = os.path.join(os.path.dirname(args.result_dir), "metric_results"   )
    print("path",path)
    combined_df = read_color_diff_files(path ,subdirectories, model_name)

    if not combined_df.empty:
        # 计算每个代次的平均子类差异值和标准差
        average_cd_per_gen = combined_df.groupby(['Gen', 'Run ID'])['Max Difference'].mean().reset_index()
        average_cd_per_gen_grouped = average_cd_per_gen.groupby('Gen')['Max Difference'].mean()
        std_cd_per_gen = average_cd_per_gen.groupby('Gen')['Max Difference'].std()

        # 绘制模型的平均子类差异值变化曲线，包括标准差阴影
        x = average_cd_per_gen_grouped.index
        y = average_cd_per_gen_grouped.values
        yerr = std_cd_per_gen.values

        # 绘制原始数据点，包括标准差阴影
        plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                     elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = slope * x + intercept

        # 绘制拟合的回归线
        plt.plot(x, regression_line, color=colors[model_idx], linestyle='-', linewidth=3,
                 label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

        plt.fill_between(x, y - yerr, y + yerr, color=colors[model_idx], alpha=0.1)

# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title('Average Subclass Difference Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(output_dir, 'average_subclass_difference_across_models.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average Subclass Difference plot for all models saved as {output_path_curve}')
 

 # 平滑函数，类似于 Tensorboard 的 smoothing 函数
def smooth(data, weight=0.75):
    scalar = data.values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return pd.Series(smoothed, index=data.index)



# 创建一个画布，用于绘制所有模型的曲线
plt.figure(figsize=(12, 8))

# 遍历每个模型，读取和绘制数据
for model_idx, model_name in enumerate(model_names):
    path = os.path.join(os.path.dirname(args.result_dir), "metric_results"   )
    print("path",path)
    combined_df = read_color_diff_files(path ,subdirectories, model_name)
 
    if not combined_df.empty:
        # 计算每个代次的平均子类差异值和标准差
        average_cd_per_gen = combined_df.groupby(['Gen', 'Run ID'])['Max Difference'].mean().reset_index()
        average_cd_per_gen_grouped = average_cd_per_gen.groupby('Gen')['Max Difference'].mean()
        std_cd_per_gen = average_cd_per_gen.groupby('Gen')['Max Difference'].std()

        # 对数据进行平滑处理
        smoothed_y = smooth(average_cd_per_gen_grouped)
        smoothed_yerr = smooth(std_cd_per_gen)

        # 绘制平滑后的数据点，包括标准差阴影
        x = average_cd_per_gen_grouped.index
        plt.errorbar(x, smoothed_y, yerr=smoothed_yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                     elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = linregress(x, smoothed_y)
        regression_line = slope * x + intercept

        # 绘制拟合的回归线
        plt.plot(x, regression_line, color=colors[model_idx], linestyle='-', linewidth=3,
                 label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

        # 绘制平滑后的标准差阴影
        plt.fill_between(x, smoothed_y - smoothed_yerr, smoothed_y + smoothed_yerr, color=colors[model_idx], alpha=0.15)

# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average Subclass Difference Value')
plt.title('Average Subclass Difference Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(output_dir, 'average_subclass_difference_across_models2.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average Subclass Difference plot for all models saved as {output_path_curve}')