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




def calculate_disparate_impact_for_all(df):
    results = []
    digits = df['Predicted Label'].unique()  # 获取所有可能的数字
    colors = df['Color'].unique()  # 获取所有可能的颜色

    for digit in digits:
        for color in colors:
            # 当前颜色为此色，其他颜色为其他
            # base_rate_current: 计算当前颜色和数字组合的基础率，即当前颜色中预测为该数字的比例。
            # base_rate_others: 计算不是当前颜色但预测为该数字的其他颜色的基础率。
            base_rate_current = (df[df['Color'] == color]['Predicted Label'] == digit).mean()
            base_rate_others = (df[df['Color'] != color]['Predicted Label'] == digit).mean()

            # 计算差异影响，并避免除以零
            if base_rate_others > 0:
                # disparate_impact = min(base_rate_current / base_rate_others, base_rate_others / base_rate_current)
                disparate_impact =  base_rate_others / base_rate_current 
            else:
                disparate_impact = None  # 如果基础率为零，则差异影响无法计算

            results.append({
                'Digit': digit,
                'Color': color,
                'Disparate Impact': disparate_impact
            })

    return results

 
results = calculate_disparate_impact_for_all(df)
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(args.result_save_path,'disparate_impact_results.csv'), index=False)
