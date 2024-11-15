import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--output_path', type=str, required=True, help='Path to save resulting plots')
args = parser.parse_args()

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['SimpleNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']  # 用于绘图标签

# 定义颜色映射
color_map = {0: 'Red', 1: 'Blue', 2: 'Green'}

# 原始化数据
num_positions = 3  # 最差到第三差
num_files = 11     # 每个模型对应 11 个CSV文件
labels = ['1st Worst', '2nd Worst', '3rd Worst']
colors = ['blue', 'green', 'red']
markers = ['o', 's', 'D']  # 不同的标记符号

# 迭代模型，设置模型子路径
all_models_worst_accuracies_mean = []
all_models_worst_accuracies_std = []

for model_idx, model_name in enumerate(model_names):
  
    # 初始化数组
    worst_accuracies_mean = np.zeros((num_positions, num_files))
    worst_accuracies_std = np.zeros((num_positions, num_files))
    
    # 迭代每个代次文件夹
    for subdir in range(11):
        csvpath = os.path.join(args.result_dir, model_name, f'gen{subdir}')
        csv_file_path = os.path.join(csvpath, 'all_images_results.csv')
        df = pd.read_csv(csv_file_path)
    
        # 获取 Run IDs
        run_ids = df['Run ID'].unique()
        worst_accuracies_runs = [[] for _ in range(num_positions)]

        # 处理每个运行 ID
        for run_id in run_ids:
            df_run = df[df['Run ID'] == run_id]
            
            # 计算每个运行 ID 的平均准确率
            subclass_accuracies = df_run.groupby('True Label').apply(lambda x: (x['True Label'] == x['Predicted Label']).mean()).values
            sorted_accuracies = sorted(subclass_accuracies)
            for pos in range(num_positions):
                worst_accuracies_runs[pos].append(sorted_accuracies[pos])

        # 计算每个位置的准确率平均值和标准差
        for pos in range(num_positions):
            accuracies = worst_accuracies_runs[pos]
            worst_accuracies_mean[pos, subdir] = np.nanmean(accuracies)
            worst_accuracies_std[pos, subdir] = np.nanstd(accuracies)

    # 保存每个模型的平均值和标准差
    all_models_worst_accuracies_mean.append(worst_accuracies_mean)
    all_models_worst_accuracies_std.append(worst_accuracies_std)

# 现在使用 all_models_worst_accuracies_mean 和 all_models_worst_accuracies_std 进行绘图
fig, axes = plt.subplots(1, len(model_names), figsize=(20, 5), sharey=True)
fig.suptitle("Trend of Worst to Third Worst Accuracies Across Generations")

# 迭代每个模型
for model_idx, ax in enumerate(axes):
    model_name = model_names[model_idx]
    display_name = display_names[model_idx]
    worst_accuracies_mean = all_models_worst_accuracies_mean[model_idx]
    worst_accuracies_std = all_models_worst_accuracies_std[model_idx]

    # 绘制每个位置的曲线和标准差阴影
    for i, label in enumerate(labels):
        x = np.arange(num_files)
        y = worst_accuracies_mean[i]
        yerr = worst_accuracies_std[i]
        
        # 绘制原始数据点
        ax.scatter(x, y, color=colors[i], s=30, alpha=0.7, marker=markers[i], label=label)

        # 绘制拟合的回归线
        slope, intercept = np.polyfit(x, y, 1)
        fit_line = slope * x + intercept
        ax.plot(x, fit_line, color=colors[i], linestyle='--', linewidth=2)
        
        # 绘制标准差阴影
        ax.fill_between(x, y - yerr, y + yerr, color=colors[i], alpha=0.1)
        
        # 设置x轴网格线隔一个单位画一条
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔1个x轴值画一个网格线

        # 设置网格线样式
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

    # 设置子图标题和标签
    ax.set_title(display_name)
    ax.set_xlabel('Generation')
    ax.set_xticks(x)

axes[0].set_ylabel('Accuracy')
# 创建自定义的图例句柄，以确保图例与标记符号和颜色一致
legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=labels[i],
                          markerfacecolor=colors[i], markersize=8, linestyle='--')
                   for i in range(num_positions)]

# 在图的顶部添加自定义的全局图例
fig.legend(handles=legend_elements, loc='upper center', ncol=num_positions, bbox_to_anchor=(0.5, 1.15), frameon=False)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 调整图与标题和图例的距离

# 保存图像
output_plot_path = os.path.join(args.output_path, 'subclass_acc_trends.png')
os.makedirs(args.output_path, exist_ok=True)
plt.savefig(output_plot_path, bbox_inches='tight')
plt.show()

print(f"Image saved in {output_plot_path}")
