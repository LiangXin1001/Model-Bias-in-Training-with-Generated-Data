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



# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--test_csv', type=str, required=True, help='CSV file path for testing dataset')
 
parser.add_argument('--test_images_dir', type=str, required=True, help='Directory path for testing images')
   
args = parser.parse_args()

  
mean_rgb , std_rgb = utils.get_mean_std(args.test_csv, [args.test_images_dir])

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean_rgb, std = std_rgb)  # 根据你的设置调整
])
# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.annotations.iloc[idx, 1]
        color = self.annotations.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, label, color
 
 
 
test_dataset = CustomDataset(csv_file= args.test_csv, root_dir = args.test_images_dir, transform = transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=False).to(device)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10).to(device)

 
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels,color in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

 
accuracy = calculate_accuracy(test_loader, model)
print(f'Accuracy of the untrained model on the test images: {accuracy:.2f}%')
