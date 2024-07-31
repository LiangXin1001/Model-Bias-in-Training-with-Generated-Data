import shutil   
import torch
from torchvision.models import resnet50
from torch import nn
import os
import pandas as pd
from PIL import Image
import sys
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
 
from torchvision import transforms
from torch.utils.data import DataLoader
import seaborn as sns
import argparse

sys.path.append(os.path.abspath( "/local/scratch/hcui25/Project/xin/CS/GAN/classify"))

from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--result_csv', type=str, required=True, help='CSV file path for testing dataset')
parser.add_argument('--result_save_path', type=str, required=True, help='CSV file path for testing dataset')
 
args = parser.parse_args()
 
df = pd.read_csv(args.result_csv)
print(df.head())


# 定义函数计算TPR和FPR
def calculate_tpr_fpr(df):
    results = []
    # 对每个数字和颜色组合进行操作
    for digit in range(10):
        for color in range(3):
            subset = df[(df['True Label'] == digit) & (df['Color'] == color)]
            TP = ((subset['True Label'] == subset['Predicted Label']) & (subset['True Label'] == digit)).sum()
            FN = ((subset['True Label'] == digit) & (subset['Predicted Label'] != digit)).sum()
            FP = ((df['True Label'] != digit) & (df['Predicted Label'] == digit) & (df['Color'] == color)).sum()
            TN = ((df['True Label'] != digit) & (df['Predicted Label'] != digit) & (df['Color'] == color)).sum()

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            
            print(f"For digit {digit}, color {color}: TP={TP}, FN={FN}, FP={FP}, TN={TN}")
 
            # results.append({'Digit': digit, 'Color': color, 'TPR': TPR, 'FPR': FPR})
            results.append({
                'Digit': digit,
                'Color': color,
                'TP': TP,
                'FN': FN,
                'FP': FP,
                'TN': TN,
                'TPR': TPR,
                'FPR': FPR
            })     
    return results


tpr_fpr_results = calculate_tpr_fpr(df)
 
 

def difference_in_means_by_color(df):
    results = {}
    for digit in range(10):
        color_accuracies = []
        for color in range(3):
            subset = df[(df['True Label'] == digit) & (df['Color'] == color)]
            accuracy = (subset['True Label'] == subset['Predicted Label']).mean()
            color_accuracies.append(accuracy)
        
        # 计算并存储每种颜色的准确率差异
        if len(color_accuracies) > 1:
            max_diff = max(color_accuracies) - min(color_accuracies)
            results[f'Digit {digit} Color Difference'] = max_diff
    
    return results


def difference_in_means_by_digit(df):
    results = {}
    for color in range(3):
        digit_accuracies = []
        for digit in range(10):
            subset = df[(df['True Label'] == digit) & (df['Color'] == color)]
            accuracy = (subset['True Label'] == subset['Predicted Label']).mean()
            digit_accuracies.append(accuracy)
        
        # 计算并存储每个数字的准确率差异
        if len(digit_accuracies) > 1:
            max_diff = max(digit_accuracies) - min(digit_accuracies)
            results[f'Color {color} Digit Difference'] = max_diff
    
    return results



color_difference_results = difference_in_means_by_color(df)
digit_difference_results = difference_in_means_by_digit(df)

print("Difference in Means by Color:")
for key, value in color_difference_results.items():
    print(f"{key}: {value}")

print("\nDifference in Means by Digit:")
for key, value in digit_difference_results.items():
    print(f"{key}: {value}")

 
tpr_fpr_df = pd.DataFrame(tpr_fpr_results)
print("TPR and FPR by digit and color:")
for index, row in tpr_fpr_df.iterrows():
    print(f"Digit {row['Digit']} - Color {row['Color']}: TPR = {row['TPR']}, FPR = {row['FPR']}")
color_diff_df = pd.DataFrame.from_dict(color_difference_results, orient='index')
digit_diff_df = pd.DataFrame.from_dict(digit_difference_results, orient='index')

# 保存到CSV
tpr_fpr_df.to_csv(os.path.join(args.result_save_path,'tpr_fpr_results.csv'))
color_diff_df.to_csv(os.path.join(args.result_save_path,'color_difference_results.csv'))
digit_diff_df.to_csv(os.path.join(args.result_save_path,'digit_difference_results.csv'))