import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score
 
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


 

# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--train_csv', type=str, required=True, help='CSV file path for training dataset')
parser.add_argument('--test_csv', type=str, required=True, help='CSV file path for testing dataset')
parser.add_argument('--train_images_dir', type=str, required=True, help='Directory path for training images')
parser.add_argument('--test_images_dir', type=str, required=True, help='Directory path for testing images')
parser.add_argument('--model_save_path', type=str, required=True, help='Directory path to save models')
parser.add_argument('--result_save_path', type=str, required=True, help='Directory path to save results')
parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')

args = parser.parse_args()

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 配置文件和图像路径
csv_file =  args.train_csv
image_folder = args.train_images_dir.split(',')

mean_rgb , std_rgb = utils.get_mean_std(csv_file, image_folder)

# 数据转换，适用于彩色图像
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean = mean_rgb, std = std_rgb)  
])
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes, device):
    if model_name.lower() == 'alexnet':
        model = alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.lower() == 'vgg19':
        model = vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.lower() == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'mobilenetv3':
        model = mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    elif model_name.lower() == 'inceptionv4':
        # 使用 InceptionV3 作为 InceptionV4 的替代
        model = inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError("Unsupported model name")
    
    return model.to(device)


class CustomDataset(Dataset):
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
        return image, label, color  # 返回图像、数值标签和颜色标签


# 加载数据集
train_csv  = args.train_csv
test_csv =  args.test_csv
dataset_path =  args.train_images_dir.split(',')
testdatapath =  args.test_images_dir
train_dataset = CustomDataset(csv_file=train_csv, img_dirs=dataset_path, transform=transform)
test_dataset = CustomDataset(csv_file=test_csv, img_dirs=[testdatapath], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

 
model = get_model(args.model_name, 10, device)
 
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
 
def train(epochs):
    model.train()
 
    for batch_idx, (data, target,_) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                

 
# 保存模型函数
def save_model(model, model_name, epoch, path):
    os.makedirs(path, exist_ok=True)
    model_filename = f"{model_name}_epoch{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(path, model_filename))
    print(f"Model saved: {model_filename}")

# 训练和测试模型
for epoch in range(1, args.epochs + 1):
    train(epoch)
    
save_model(model, args.model_name, args.epochs, args.model_save_path)


