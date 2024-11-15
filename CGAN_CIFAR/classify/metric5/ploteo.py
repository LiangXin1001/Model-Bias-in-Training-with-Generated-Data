import pandas as pd
import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate ovrEO for each Superclass')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--output_path', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']  # 用于绘图标签


base_dir = os.path.dirname(args.result_dir)

# 图形样式设置
markers = ['o', 's', 'D', '^', 'v']  # 标记符号
colors = ['blue', 'green', 'red', 'purple', 'orange']  # 每个模型的颜色

# 创建一个画布，用于绘制所有模型
plt.figure(figsize=(12, 8))

# 遍历每个模型
for model_idx, model_name in enumerate(model_names):
     
    save_path = os.path.join(base_dir,'metric_results',  model_name)
  
    all_eo_results = []

    # 读取每个代次的 EO 结果
    for i in range(11):
        eo_csv_path = os.path.join(save_path, f'eo_results_{i}.csv')
        if os.path.exists(eo_csv_path):
            eo_df = pd.read_csv(eo_csv_path)
            eo_df['Generation'] = i  # 添加生成代次信息
            all_eo_results.append(eo_df)

    # 合并所有代次的 EO 数据
    combined_df = pd.concat(all_eo_results, ignore_index=True)

    # 计算每个代次的平均 EO 值和标准差
    average_eo_per_gen = combined_df.groupby('Generation')['EO'].mean()
    std_eo_per_gen = combined_df.groupby('Generation')['EO'].std()

    # 绘制模型的平均 EO 值变化曲线，包括标准差阴影和拟合回归线
    x = average_eo_per_gen.index
    y = average_eo_per_gen.values
    yerr = std_eo_per_gen.values

    # 绘制原始数据点
    plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                 elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

    # 计算线性回归
    slope, intercept = np.polyfit(x, y, 1)
    fit_line = slope * x + intercept

    # 绘制拟合的回归线
    plt.plot(x, fit_line, color=colors[model_idx], linestyle='-', linewidth=3, label=f'{display_names[model_idx]} Regression')
    plt.fill_between(x, y - yerr, y + yerr, color=colors[model_idx], alpha=0.05)


# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(args.output_path, 'average_eo_values_across_models.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average EO values plot for all models saved as {output_path_curve}')



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
    save_path = os.path.join(base_dir, 'metric_results', model_name)
    all_eo_results = []

    # 读取每个代次的 EO 结果
    for i in range(11):
        eo_csv_path = os.path.join(save_path, f'eo_results_{i}.csv')
        if os.path.exists(eo_csv_path):
            eo_df = pd.read_csv(eo_csv_path)
            eo_df['Generation'] = i  # 添加生成代次信息
            all_eo_results.append(eo_df)

    # 合并所有代次的 EO 数据
    combined_df = pd.concat(all_eo_results, ignore_index=True)

    # 计算每个代次的平均 EO 值和标准差
    average_eo_per_gen = combined_df.groupby('Generation')['EO'].mean()
    std_eo_per_gen = combined_df.groupby('Generation')['EO'].std()

    # 对数据进行平滑处理
    smoothed_y = smooth(average_eo_per_gen)
    smoothed_yerr = smooth(std_eo_per_gen)

    # 绘制平滑后的数据点，包括标准差阴影
    x = average_eo_per_gen.index
    plt.errorbar(x, smoothed_y, yerr=smoothed_yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                 elinewidth=2, capsize=4, label=f'{display_names[model_idx]} Data')

    # 计算线性回归并绘制拟合的回归线
    slope, intercept = np.polyfit(x, smoothed_y, 1)
    fit_line = slope * x + intercept

    plt.plot(x, fit_line, color=colors[model_idx], linestyle='-', linewidth=3, label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

    # 绘制平滑后的标准差阴影
    plt.fill_between(x, smoothed_y - smoothed_yerr, smoothed_y + smoothed_yerr, color=colors[model_idx], alpha=0.05)

# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Average EO Value')
plt.title('Average EO Values Across Generations for Different Models')
plt.xticks(range(11))  # 确保 x 轴显示所有代次值
plt.grid(True)
plt.legend(title='Model')
plt.tight_layout()

# 保存图像
output_path_curve = os.path.join(args.output_path, 'average_eo_values_across_models3.png')
plt.savefig(output_path_curve, bbox_inches='tight')
plt.show()

print(f'Average EO values plot for all models saved as {output_path_curve}')
# # 设置 Seaborn 样式
# sns.set(style="whitegrid")

# # 创建一个画布，用于绘制所有模型
# plt.figure(figsize=(12, 8))

# # 遍历每个模型
# for model_idx, model_name in enumerate(model_names):
#     save_path = os.path.join(base_dir, 'metric_results', model_name)
  
#     all_eo_results = []

#     # 读取每个代次的 EO 结果
#     for i in range(11):
#         eo_csv_path = os.path.join(save_path, f'eo_results_{i}.csv')
#         if os.path.exists(eo_csv_path):
#             eo_df = pd.read_csv(eo_csv_path)
#             eo_df['Generation'] = i  # 添加生成代次信息
#             all_eo_results.append(eo_df)

#     # 合并所有代次的 EO 数据
#     combined_df = pd.concat(all_eo_results, ignore_index=True)

#     # 计算每个代次的平均 EO 值和标准差
#     combined_df['Model'] = display_names[model_idx]  # 添加模型名称列，用于绘图
#     combined_df['Color'] = colors[model_idx]  # 添加颜色列，用于绘图

#     # 绘制带置信区间的平滑曲线
#     sns.lineplot(
#         data=combined_df,
#         x='Generation',
#         y='EO',
#         label=f'{display_names[model_idx]} Data',
#         color=colors[model_idx],
#         errorbar='sd',  # 使用标准差作为置信区间
#         linewidth=1,
#         alpha=0.01 # 明确设置曲线的透明度
#     )

#     # 绘制原始数据点
#     average_eo_per_gen = combined_df.groupby('Generation')['EO'].mean()
#     std_eo_per_gen = combined_df.groupby('Generation')['EO'].std()

#     x = average_eo_per_gen.index
#     y = average_eo_per_gen.values
#     yerr = std_eo_per_gen.values

#     plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
#                  elinewidth=2, capsize=4, alpha=0.8)

#     # 计算线性回归
#     slope, intercept = np.polyfit(x, y, 1)
#     fit_line = slope * x + intercept

#     # 绘制拟合的回归线
#     plt.plot(x, fit_line, color=colors[model_idx], linestyle='-', linewidth=3, label=f'{display_names[model_idx]} Regression (y={slope:.4f}x + {intercept:.4f})')

# # 设置图表属性
# plt.xlabel('Generation')
# plt.ylabel('Average EO Value')
# plt.title('Average EO Values Across Generations for Different Models')
# plt.xticks(range(11))  # 确保 x 轴显示所有代次值
# plt.grid(True)
# plt.legend(title='Model', loc='best')
# plt.tight_layout()

# # 保存图像
# output_path_curve = os.path.join(args.output_path, 'average_eo_values_across_models2.png')
# plt.savefig(output_path_curve, bbox_inches='tight')
# plt.show()

# print(f'Average EO values plot for all models saved as {output_path_curve}')