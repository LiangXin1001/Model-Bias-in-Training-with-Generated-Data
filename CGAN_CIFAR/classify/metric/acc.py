
import pandas as pd
 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import sys
import os
 
current_directory = os.path.dirname(__file__)  
parent_directory = os.path.dirname(current_directory)  
sys.path.append(parent_directory)
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf ,CIFAR_100_CLASS_MAP,generate_full_subclass_map
 

csv_file_path = 'results/resnet50/test_results_0.csv'  # 替换为你的 CSV 文件路径

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

overall_accuracy = accuracy_score(df['True Superclass'], df['Predicted Superclass'])
print(f"Overall Accuracy: {overall_accuracy:.2%}")

# 计算每个大类的正确率
superclass_accuracies = df.groupby('True Superclass').apply(
    lambda x: accuracy_score(x['True Superclass'], x['Predicted Superclass'])
)
# print("\nAccuracy per Superclass:")
# print(superclass_accuracies)

# 计算每个子类的正确率
subclass_accuracies = df.groupby('True Subclass').apply(
    lambda x: accuracy_score(x['True Superclass'], x['Predicted Superclass'])   
)
# print("\nAccuracy per Subclass:")
# print(subclass_accuracies)

worst_accuracies = []
second_worst_accuracies = []
third_worst_accuracies = []
fourth_worst_accuracies = []
fifth_worst_accuracies = []
# 计算每个超类的子类正确率
for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
    print(f"Accuracy for subclasses in superclass {superclass}:")
    subclass_accuracies = []
    for subclass in subclasses:
        # 过滤出当前子类的所有数据
        subclass_data = df[df['True Subclass Name'] == subclass]
        accuracy = accuracy_score(subclass_data['True Superclass'], subclass_data['Predicted Superclass'])
        subclass_accuracies.append(accuracy)
        print(f"{subclass}: {accuracy:.2%}")

    # 对子类准确率进行排序
    sorted_accuracies = sorted(subclass_accuracies)

    # 收集每个超类中最差到第五差的准确率
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
 