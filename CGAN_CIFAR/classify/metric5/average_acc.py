import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import seaborn as sns  
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import sys
from matplotlib.lines import Line2D 
import matplotlib.pyplot as plt
current_directory = os.path.dirname(__file__)  
parent_directory = os.path.dirname(current_directory)  
sys.path.append(parent_directory)
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf ,CIFAR_100_CLASS_MAP,generate_full_subclass_map
import argparse
 
parser = argparse.ArgumentParser(description='Calculate Disparate Impact (DI) for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
 
parser.add_argument('--output_path', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()
  


base_dir = os.path.dirname(args.result_dir)
# 定义模型名称和绘图显示名称
 
 
num_positions = 5  # 最差到第五差
num_files = 11     # 每个模型对应11个CSV文件（从0到10）
labels = ['Worst', '2nd Worst', '3rd Worst', '4th Worst', '5th Worst']
colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', 'D', '^', 'v']  # 不同的标记符号

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']  # 用于绘图标签

# 初始化数据存储
all_models_mean_accuracies = []
all_models_std_accuracies = []

# 遍历模型，计算每个模型的平均准确率
for model_name in model_names:
    csvpath = os.path.join(base_dir, model_name)
    csv_files = [f'test_results_{i}.csv' for i in range(11)]
    
    # 存储每一代的平均准确率
    mean_accuracies = []
    std_accuracies = []
    
    # 遍历每个文件
    for csv_file in csv_files:
        csv_file_path = os.path.join(csvpath, csv_file)
        df = pd.read_csv(csv_file_path, header=None)

        # 设置列名并删除重复表头行
        column_names = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID', 'True Superclass Name', 'True Subclass Name']
        df.columns = column_names
        df = df[df['Image'] != 'Image'].reset_index(drop=True)  # 去除含表头内容的行

        # 将数值列转换为整数类型
        numeric_columns = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').astype(int)

        # 获取 Run IDs 并计算每个 Run ID 的准确率
        run_ids = df['Run ID'].unique()
        accuracies = []

        for run_id in run_ids:
            df_run = df[df['Run ID'] == run_id]
            accuracy = accuracy_score(df_run['True Superclass'], df_run['Predicted Superclass'])
            accuracies.append(accuracy)

        # 计算每一代的平均准确率和标准差
        mean_accuracies.append(np.mean(accuracies))
        std_accuracies.append(np.std(accuracies))

    # 存储每个模型的平均准确率和标准差
    all_models_mean_accuracies.append(mean_accuracies)
    all_models_std_accuracies.append(std_accuracies)

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Average Accuracy", y=1.03)

csv_file_indices = np.arange(len(csv_files))

# 遍历每个模型进行绘图
for model_idx, model_name in enumerate(model_names):
    mean_accuracies = all_models_mean_accuracies[model_idx]
    std_accuracies = all_models_std_accuracies[model_idx]

    # 绘制原始数据点
    ax.scatter(csv_file_indices, mean_accuracies, color=colors[model_idx], s=30, alpha=0.9, marker=markers[model_idx], label=f'{display_names[model_idx]} Data')

    # 拟合回归线
    slope, intercept = np.polyfit(csv_file_indices, mean_accuracies, 1)
    fit_line = slope * csv_file_indices + intercept
    ax.plot(csv_file_indices, fit_line, color=colors[model_idx], linestyle='-', linewidth=3, label=f'{display_names[model_idx]} Regression')

    # 绘制标准差阴影
    ax.fill_between(csv_file_indices, np.array(mean_accuracies) - np.array(std_accuracies), 
                    np.array(mean_accuracies) + np.array(std_accuracies), color=colors[model_idx], alpha=0.1)

# 设置图表属性
ax.set_xlabel('Generation')
ax.set_ylabel('Average Accuracy')
ax.set_xticks(csv_file_indices)
ax.grid(True)

# 创建自定义图例
legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=f'{display_names[i]}',
                          markerfacecolor=colors[i], markersize=8, linestyle='-')
                   for i in range(len(model_names))]
# legend_elements += [Line2D([0], [0], color=colors[i], linestyle='-', linewidth=3, label=f'{display_names[i]} Regression')
#                     for i in range(len(model_names))]

# 在图的顶部添加自定义的全局图例
fig.legend(handles=legend_elements, loc='upper center', ncol=len(model_names), bbox_to_anchor=(0.5, 1.08), frameon=False)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=1)

# 保存图像
output_plot_path = os.path.join(args.output_path, 'average_accuracy_trends_across_models.png')
plt.savefig(output_plot_path, bbox_inches='tight')
 
print(f"Image saved in {output_plot_path}")