# MNIST image generation using Conditional DCGAN
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
 
import argparse
import pandas as pd
from PIL import Image
 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径,包含图片名称和标签。
            img_dirs (string): 包含所有图片的目录路径。
            transform (callable, optional): 可选的转换函数，应用于样本。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        image = None
        # 在所有目录中查找图像
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in any of the directories.")
        
        label = self.img_labels.iloc[idx, 1]
        color = self.img_labels.iloc[idx, 2]  # 加载颜色标签
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)  # 默认转换为张量
        return image, label  # 返回图像、数值标签和颜色标签

