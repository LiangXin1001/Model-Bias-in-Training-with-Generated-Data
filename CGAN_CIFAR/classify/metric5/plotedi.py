import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--output_path', type=str, required=True, help='Path to save resulting plots')
args = parser.parse_args()

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']

# 设置基本路径
base_dir = os.path.dirname(args.result_dir)
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)
subdirectories = [str(i) for i in range(11)]

# 读取子类差异值文件的函数
def read_di_files(base_path, subdirectories, model_name):
    dfs = []
   
    for subdir in subdirectories:
        filepath = os.path.join(base_path, "metric_results",model_name, f'disparate_impact_results_{subdir}.csv')
        if os.path.exists(filepath):
            print(".path.exists(filepath) ")
            df = pd.read_csv(filepath)
            df['Gen'] = int(subdir)  # 添加生成代次信息
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        if combined_df.empty:
            print(f"No data available for model {model_name}")
        print("combined_df",combined_df)
        return combined_df
    else:
        print(f"No data available for model {model_name}")
        return pd.DataFrame()  # 返回空的 DataFrame 如果没有文件

# 图形样式设置
markers = ['o', 's', 'D', '^', 'v']  # 标记符号
colors = ['blue', 'green', 'red', 'purple', 'orange']  # 每个模型的颜色

# 创建一个画布，用于绘制所有模型
plt.figure(figsize=(12, 8))

# 遍历每个模型
for model_idx, model_name in enumerate(model_names):
    combined_df = read_di_files(base_dir, subdirectories, model_name)

    if not combined_df.empty:
        # 计算每个代次的平均差异影响值和标准差
        average_di_per_gen = combined_df.groupby(['Gen', 'Run ID'])['Disparate Impact'].mean().reset_index()
        average_di_per_gen_grouped = average_di_per_gen.groupby('Gen')['Disparate Impact'].mean()
        std_di_per_gen = average_di_per_gen.groupby('Gen')['Disparate Impact'].std()

        # 绘制模型的平均差异影响值变化曲线，包括标准差阴影
        x = average_di_per_gen_grouped.index
        y = average_di_per_gen_grouped.values
        yerr = std_di_per_gen.values

        # 绘制原始数据点，包括标准差阴影
        plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                     elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

        # 计算线性回归
        slope, intercept = np.polyfit(x, y, 1)
        fit_line = slope * x + intercept

        # 绘制拟合的回归线
        plt.plot(x, fit_line, color=colors[model_idx], linestyle='-', linewidth=3,
                 label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

        plt.fill_between(x, y - yerr, y + yerr, color=colors[model_idx], alpha=0.05)

# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(output_dir, 'average_disparate_impact_across_models.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average Disparate Impact values plot for all models saved as {output_path_curve}')



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

# 创建一个画布，用于绘制所有模型
plt.figure(figsize=(12, 8))

# 遍历每个模型
for model_idx, model_name in enumerate(model_names):
    combined_df = read_di_files(base_dir, subdirectories, model_name)

    if not combined_df.empty:
        # 计算每个代次的平均差异影响值和标准差
        average_di_per_gen = combined_df.groupby(['Gen', 'Run ID'])['Disparate Impact'].mean().reset_index()
        average_di_per_gen_grouped = average_di_per_gen.groupby('Gen')['Disparate Impact'].mean()
        std_di_per_gen = average_di_per_gen.groupby('Gen')['Disparate Impact'].std()

        # 对数据进行平滑处理
        smoothed_y = smooth(average_di_per_gen_grouped)
        smoothed_yerr = smooth(std_di_per_gen)

        # 绘制平滑后的数据点，包括标准差阴影
        x = average_di_per_gen_grouped.index
        plt.errorbar(x, smoothed_y, yerr=smoothed_yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                     elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

        # 计算线性回归并绘制拟合的回归线
        slope, intercept = np.polyfit(x, smoothed_y, 1)
        fit_line = slope * x + intercept

        plt.plot(x, fit_line, color=colors[model_idx], linestyle='-', linewidth=3,
                 label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

        # 绘制平滑后的标准差阴影
        plt.fill_between(x, smoothed_y - smoothed_yerr, smoothed_y + smoothed_yerr, color=colors[model_idx], alpha=0.05)

# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average Disparate Impact Value')
plt.title('Average Disparate Impact Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(output_dir, 'average_disparate_impact_across_models3.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average Disparate Impact values plot for all models saved as {output_path_curve}')

# # 创建一个画布，用于绘制所有模型
# plt.figure(figsize=(12, 8))

# # 遍历每个模型
# for model_idx, model_name in enumerate(model_names):
#     combined_df = read_di_files(base_dir, subdirectories, model_name)

#     if not combined_df.empty:
#         # 计算每个代次的平均差异影响值和标准差
#         average_di_per_gen = combined_df.groupby(['Gen', 'Run ID'])['Disparate Impact'].mean().reset_index()
#         average_di_per_gen_grouped = average_di_per_gen.groupby('Gen')['Disparate Impact'].mean()
#         std_di_per_gen = average_di_per_gen.groupby('Gen')['Disparate Impact'].std()

#         # 将数据转换为绘图输入格式
#         plot_data = pd.DataFrame({
#             'Generation': average_di_per_gen_grouped.index,
#             'Disparate Impact': average_di_per_gen_grouped.values,
#             'Model': display_names[model_idx]
#         })

#         # 使用 Seaborn 绘制平滑的标准差阴影和平均曲线
#         sns.lineplot(
#             x='Generation',
#             y='Disparate Impact',
#             data=plot_data,
#             color=colors[model_idx],
#             label=f'{display_names[model_idx]} Data',
#             marker=markers[model_idx],
#             linewidth=2.5,
#             alpha=0.1,
#             errorbar='sd'  # 标准差作为误差条，Seaborn会自动平滑阴影
#         )

#         # 计算线性回归并画出回归直线
#         x = average_di_per_gen_grouped.index
#         y = average_di_per_gen_grouped.values
#         slope, intercept = np.polyfit(x, y, 1)
#         regression_line = slope * x + intercept

#         # 绘制拟合的回归线
#         plt.plot(
#             x, regression_line,
#             color=colors[model_idx], linestyle='-', linewidth=2.5,
#             label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})'
#         )

# # 设置图表属性
# plt.xlabel('Generation')
# plt.ylabel('Average Disparate Impact Value')
# plt.title('Average Disparate Impact Across Generations for Different Models')
# plt.xticks(range(11))  # 确保 x 轴显示所有代次值
# plt.grid(True)
# plt.legend(title='Model')
# plt.tight_layout()

# # 保存图像
# output_path_curve = os.path.join(output_dir, 'average_disparate_impact_across_models2.png')
# plt.savefig(output_path_curve, bbox_inches='tight')
 
# print(f'Average Disparate Impact values plot for all models saved as {output_path_curve}')