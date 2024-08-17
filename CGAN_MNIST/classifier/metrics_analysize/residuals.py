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
sys.path.insert(0, os.path.abspath('..'))  
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


# use binary residuals
def calculate_binary_residuals_for_all_colors(df):
    results = {}
    colors = df['Color'].unique()  # 获取所有唯一的颜色标识

    # 为每种颜色与其他颜色比较计算正确率差异
    for color in colors:
        group1 = df[df['Color'] == color]
        group2 = df[df['Color'] != color]

        if not group1.empty and not group2.empty:
            correctness_mean_g1 = group1['Correctness'].mean()
            correctness_mean_g2 = group2['Correctness'].mean()

            # 计算两个群组之间预测正确率的差异
            difference = abs(correctness_mean_g1 - correctness_mean_g2)
            results[f'Difference between Color {color} and Others'] = difference

    return results

 
df['Correctness'] = (df['True Label'] == df['Predicted Label']).astype(int)
 

color_differences = calculate_binary_residuals_for_all_colors(df)
for key, value in color_differences.items():
    print(f"{key}: {value}")

results_df = pd.DataFrame(list(color_differences.items()), columns=['Color', 'Difference in Correctness'])
 
results_df.to_csv(os.path.join(args.result_save_path,'color_differences.csv'), index=False)