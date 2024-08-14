
import pandas as pd
 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import sys
import os
import matplotlib.pyplot as plt
current_directory = os.path.dirname(__file__)  
parent_directory = os.path.dirname(current_directory)  
sys.path.append(parent_directory)
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf ,CIFAR_100_CLASS_MAP,generate_full_subclass_map
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate Disparate Impact (DI) for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()


# csv_file_path = 'results/resnet50/test_results_0.csv'   

# # 读取 CSV 文件
# df = pd.read_csv(csv_file_path)

# overall_accuracy = accuracy_score(df['True Superclass'], df['Predicted Superclass'])
# print(f"Overall Accuracy: {overall_accuracy:.2%}")

# # 计算每个大类的正确率
# df['Correct Superclass Prediction'] = df['True Superclass'] == df['Predicted Superclass']

# superclass_accuracies =  df.groupby('True Superclass')['Correct Superclass Prediction'].mean()
# print("\nAccuracy per Superclass:")
# print(superclass_accuracies)

# # 计算每个子类的正确率
# df['Correct Subclass Prediction'] = (df['True Superclass'] == df['Predicted Superclass']) & (df['True Subclass'] == df['Predicted Subclass'])
# subclass_accuracies = df.groupby('True Subclass')['Correct Subclass Prediction'].mean()
# print("\nAccuracy per Subclass:")
# print(subclass_accuracies)

# # 计算每个超类的子类正确率
# for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
#     print(f"Accuracy for subclasses in superclass {superclass}:")
#     subclass_accuracies = []
#     for subclass in subclasses:
#         # 过滤出当前子类的所有数据
#         subclass_data = df[df['True Subclass Name'] == subclass]
#         accuracy = accuracy_score(subclass_data['True Superclass'], subclass_data['Predicted Superclass'])
#         subclass_accuracies.append(accuracy)
#         print(f"{subclass}: {accuracy:.2%}")

#     # 对子类准确率进行排序
#     sorted_accuracies = sorted(subclass_accuracies)

#     # 收集每个超类中最差到第五差的准确率
#     if len(sorted_accuracies) >= 5:
#         worst_accuracies.append(sorted_accuracies[0])
#         second_worst_accuracies.append(sorted_accuracies[1])
#         third_worst_accuracies.append(sorted_accuracies[2])
#         fourth_worst_accuracies.append(sorted_accuracies[3])
#         fifth_worst_accuracies.append(sorted_accuracies[4])


# print("Average of the worst subclass accuracies:", sum(worst_accuracies) / len(worst_accuracies))
# print("Average of the second worst subclass accuracies:", sum(second_worst_accuracies) / len(second_worst_accuracies))
# print("Average of the third worst subclass accuracies:", sum(third_worst_accuracies) / len(third_worst_accuracies))
# print("Average of the fourth worst subclass accuracies:", sum(fourth_worst_accuracies) / len(fourth_worst_accuracies))
# print("Average of the fifth worst subclass accuracies:", sum(fifth_worst_accuracies) / len(fifth_worst_accuracies))

 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

# 获取当前文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from utils.datasets import CIFAR_100_CLASS_MAP  # 确保这个import路径正确

# 定义CSV文件路径
base_dir = args.result_dir
csv_files = [f'test_results_{i}.csv' for i in range(11)]  # 从0到10的文件名列表
worst_accuracies_all = np.zeros((5, len(csv_files)))  # 5行（每个级别的准确率），11列（每个文件）

worst_accuracies = []
second_worst_accuracies = []
third_worst_accuracies = []
fourth_worst_accuracies = []
fifth_worst_accuracies = []
# 遍历每个文件并计算准确率
for file_index, csv_file in enumerate(csv_files):
    csv_file_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_file_path)

    # 计算整体准确率
    overall_accuracy = accuracy_score(df['True Superclass'], df['Predicted Superclass'])
    print(f"\n{csv_file} - Overall Accuracy: {overall_accuracy:.2%}")

    # 计算每个超类的子类的准确率
    for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
        print(f"Accuracy for subclasses in superclass {superclass}:")
        subclass_accuracies = []
        for subclass in subclasses:
            # 过滤出当前子类的所有数据
            subclass_data = df[df['True Subclass Name'] == subclass]
            accuracy = accuracy_score(subclass_data['True Superclass'], subclass_data['Predicted Superclass'])
            subclass_accuracies.append(accuracy)
            print(f"{subclass}: {accuracy:.2%}")

        # 将准确率排序并取最差的五个
        sorted_accuracies = sorted(subclass_accuracies)
        if len(sorted_accuracies) >= 5:
            worst_accuracies.append(sorted_accuracies[0])
            second_worst_accuracies.append(sorted_accuracies[1])
            third_worst_accuracies.append(sorted_accuracies[2])
            fourth_worst_accuracies.append(sorted_accuracies[3])
            fifth_worst_accuracies.append(sorted_accuracies[4])

        
    print("Average of the worst subclass accuracies:", sum(worst_accuracies) / len(worst_accuracies))
    print("Average of the second worst subclass accuracies:", sum(second_worst_accuracies) / len(second_worst_accuracies))
    print("Average of the third worst subclass accuracies:", sum(third_worst_accuracies) / len(third_worst_accuracies))
    print("Average of the fourth worst subclass accuracies:", sum(fourth_worst_accuracies) / len(fourth_worst_accuracies))
    print("Average of the fifth worst subclass accuracies:", sum(fifth_worst_accuracies) / len(fifth_worst_accuracies))

    worst_accuracies_all[0,file_index] = sum(worst_accuracies) / len(worst_accuracies)
    worst_accuracies_all[1,file_index] = sum(second_worst_accuracies) / len(second_worst_accuracies)
    worst_accuracies_all[2,file_index] =sum(third_worst_accuracies) / len(third_worst_accuracies)
    worst_accuracies_all[3,file_index] = sum(fourth_worst_accuracies) / len(fourth_worst_accuracies)
    worst_accuracies_all[4,file_index] =sum(fifth_worst_accuracies) / len(fifth_worst_accuracies)



# 绘制准确率趋势图
plt.figure(figsize=(12, 8))
labels = ['Worst', '2nd Worst', '3rd Worst', '4th Worst', '5th Worst']
for i in range(5):
    plt.plot(range(len(csv_files)), worst_accuracies_all[i], marker='o', label=f'{labels[i]} Accuracy')

plt.xlabel('CSV File Index')
plt.ylabel('Accuracy')
plt.title('Trend of Worst to Fifth Worst Accuracies Across CSV Files')
plt.xticks(range(len(csv_files)), [f'File {i}' for i in range(len(csv_files))], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs(f'images/{args.model_name}', exist_ok=True)
# 保存图像
output_plot_path = f'images/{args.model_name}/accuracy_trends.png'
plt.savefig(output_plot_path)
plt.show()

print(f"Plot saved as {output_plot_path}")


csv_file_indices = np.arange(len(csv_files))

plt.figure(figsize=(12, 8))
labels = ['Worst', '2nd Worst', '3rd Worst', '4th Worst', '5th Worst']
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i in range(5):
    # 计算线性回归参数
    slope, intercept = np.polyfit(csv_file_indices, worst_accuracies_all[i], 1)
    # 生成拟合直线
    fit_line = slope * csv_file_indices + intercept

    # 绘制数据点和拟合直线
    plt.scatter(csv_file_indices, worst_accuracies_all[i], color=colors[i], label=f'{labels[i]} Accuracy')
    plt.plot(csv_file_indices, fit_line, color=colors[i], label=f'{labels[i]} Fit (y={slope:.4f}x + {intercept:.4f})')

plt.xlabel('CSV File Index')
plt.ylabel('Accuracy')
plt.title('Trend of Worst to Fifth Worst Accuracies Across CSV Files with Linear Regression')
plt.xticks(csv_file_indices, [f'File {i}' for i in range(len(csv_files))], rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
output_plot_path = f'images/{args.model_name}/accuracy_trends_regression.png'
plt.savefig(output_plot_path)
plt.close()

print(f"Plot saved as {output_plot_path}")