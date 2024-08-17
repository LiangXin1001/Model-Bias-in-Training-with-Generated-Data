import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import argparse

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--model_names', type=str,  required=True, help='The names of the models to process results for')
args = parser.parse_args()
   

# Function to read all images result files with dynamic paths
def read_all_images_result_files(base_path, model_name, generations):
    dfs = []
    for gen in generations:
        result_path = f"{base_path}/{model_name}/gen{gen}/all_images_results.csv"
        df = pd.read_csv(result_path)
        df['Gen'] = f"gen{gen}"  # Add generation column
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# 假设你已经通过命令行参数获取了模型名称
model_name = args.model_names  # 替换为实际模型名称
base_dir = 'results'  # 替换为实际结果目录
generations = range(11)  # 代表从0到10的代数

# 读取并组合所有结果文件
df = read_all_images_result_files(base_dir, model_name, generations)

# 用于存储最差的准确率
worst_accuracies_all = np.zeros((3, len(generations)))  # 3行（每个级别的准确率），11列（每个文件）

# 遍历每个 generation 并计算准确率
for gen_index, gen in enumerate(generations):
    # 过滤出当前 generation 的数据
    gen_data = df[df['Gen'] == f'gen{gen}']

    # 计算每个True Label（超类）的Color（子类）准确率
    worst_accuracies = []
    second_worst_accuracies = []
    third_worst_accuracies = []
    for true_label in gen_data['True Label'].unique():
        color_accuracies = []
        for color in gen_data[gen_data['True Label'] == true_label]['Color'].unique():
            # 过滤出当前Color的所有数据
            color_data = gen_data[(gen_data['True Label'] == true_label) & (gen_data['Color'] == color)]
            accuracy = accuracy_score(color_data['True Label'], color_data['Predicted Label'])
            color_accuracies.append(accuracy)

        # 将准确率排序并取最差的三个
        sorted_accuracies = sorted(color_accuracies)
        if len(sorted_accuracies) >= 3:
            worst_accuracies.append(sorted_accuracies[0])
            second_worst_accuracies.append(sorted_accuracies[1])
            third_worst_accuracies.append(sorted_accuracies[2])

    # 计算平均最差准确率
    worst_accuracies_all[0, gen_index] = sum(worst_accuracies) / len(worst_accuracies)
    worst_accuracies_all[1, gen_index] = sum(second_worst_accuracies) / len(second_worst_accuracies)
    worst_accuracies_all[2, gen_index] = sum(third_worst_accuracies) / len(third_worst_accuracies)

# 绘制准确率趋势图
plt.figure(figsize=(12, 8))
labels = ['Worst', '2nd Worst', '3rd Worst']
for i in range(3):
    plt.plot(range(len(generations)), worst_accuracies_all[i], marker='o', label=f'{labels[i]} Accuracy')

plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Trend of Worst to Third Worst Accuracies Across Generations')
plt.xticks(range(len(generations)), [f'gen{i}' for i in range(len(generations))], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs(f'images/{model_name}', exist_ok=True)
# 保存图像
output_plot_path = f'images/{model_name}/accuracy_trends.png'
plt.savefig(output_plot_path)
plt.show()

print(f"Plot saved as {output_plot_path}")

# 绘制带线性回归的准确率趋势图
generation_indices = np.arange(len(generations))

plt.figure(figsize=(12, 8))
labels = ['Worst', '2nd Worst', '3rd Worst']
colors = ['blue', 'green', 'red']

for i in range(3):
    # 计算线性回归参数
    slope, intercept = np.polyfit(generation_indices, worst_accuracies_all[i], 1)
    # 生成拟合直线
    fit_line = slope * generation_indices + intercept

    # 绘制数据点和拟合直线
    plt.scatter(generation_indices, worst_accuracies_all[i], color=colors[i], label=f'{labels[i]} Accuracy')
    plt.plot(generation_indices, fit_line, color=colors[i], label=f'{labels[i]} Fit (y={slope:.4f}x + {intercept:.4f})')

plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Trend of Worst to Third Worst Accuracies Across Generations with Linear Regression')
plt.xticks(generation_indices, [f'gen{i}' for i in range(len(generations))], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
output_plot_path = f'images/{model_name}/accuracy_trends_regression.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Plot saved as {output_plot_path}")
