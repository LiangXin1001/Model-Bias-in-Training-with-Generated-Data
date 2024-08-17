import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import sys
import matplotlib.pyplot as plt

# 设置当前工作目录，以便导入其他模块
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

# 命令行参数解析
import argparse
parser = argparse.ArgumentParser(description='Calculate accuracies for CSV results')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for generating results')

args = parser.parse_args()
base_path = 'results'
# 定义CSV文件路径
base_dir = os.path.join(base_path, args.model_name)
csv_files = [os.path.join(base_dir, f'gen{i}', 'all_images_results.csv') for i in range(11)]  # 假设生成了11个结果文件

# 颜色映射
color_map = {0: 'Red', 1: 'Blue', 2: 'Green'}

# 存储每种颜色的最差准确率数组
color_accuracies = {color: [] for color in color_map.values()}

# 遍历每个CSV文件计算准确率
for file_index, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    true_labels = df['True Label'].values
    predicted_labels = df['Predicted Label'].values
    color_labels = df['Color'].values

    # 按颜色分类并计算准确率
    for color, color_name in color_map.items():
        color_data = df[df['Color'] == color]
        if not color_data.empty:
            accuracy = accuracy_score(color_data['True Label'], color_data['Predicted Label'])
            color_accuracies[color_name].append(accuracy)
            print(f"{csv_file} - {color_name} Color Accuracy: {accuracy:.2%}")

# 绘制每种颜色的准确率趋势
plt.figure(figsize=(12, 8))
markers = ['o', 's', '^']  # 不同的标记符号
colors = ['red', 'blue', 'green']  # 对应颜色标记

for color_name, marker, color in zip(color_accuracies.keys(), markers, colors):
    plt.plot(color_accuracies[color_name], marker=marker, linestyle='-', color=color, label=f'{color_name} Accuracy')

plt.xlabel('CSV File Index')
plt.ylabel('Accuracy')
plt.title('Trend of Color Accuracies Across CSV Files')
plt.xticks(range(len(csv_files)), [f'Gen {i}' for i in range(len(csv_files))], rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像
output_path = f'images/{args.model_name}/color_accuracies_trends.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

print(f"Plot saved as {output_path}")
