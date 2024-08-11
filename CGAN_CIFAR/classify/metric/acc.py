
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

# 计算每个超类的子类正确率
for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
    print(f"Accuracy for subclasses in superclass {superclass}:")
    for subclass in subclasses:
        # 过滤出当前子类的所有数据
        subclass_data = df[df['True Subclass Name'] == subclass]
 
        accuracy = accuracy_score(subclass_data['True Superclass'], subclass_data['Predicted Superclass'])
        print(f"{subclass}: {accuracy:.2%}")




# 生成并打印分类报告
# print("\nClassification Report for Superclasses:")
# print(classification_report(df['True Superclass'], df['Predicted Superclass']))

# print("\nClassification Report for Subclasses:")
# print(classification_report(df['True Subclass'], df['Predicted Superclass']))  # 注意这里可能需要调整