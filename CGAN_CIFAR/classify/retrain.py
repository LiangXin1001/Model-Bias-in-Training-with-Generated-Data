import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from utils.datasets import SuperCIFAR100, GeneratedDataset 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.transforms as transforms
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
from torch.utils.data import ConcatDataset, DataLoader 
import os
 
from utils.datasets import SuperCIFAR100, GeneratedDataset
# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier with custom dataset')
    parser.add_argument('--gennum', type=int, required=True, help='Generator number for filename customization')
    parser.add_argument('--data_root_paths', type=str,  default= "",help='Directory path to save models')
    parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_arguments()
# 加载模型
def load_model(model_path, num_classes, device):
    model = resnet50(pretrained=False)  # 确保模型架构与保存的模型一致
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 确保最后一层是适应您的类别数
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model
tf = transforms.Compose([transforms.Resize(64),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

# 数据加载和预处理
trainset = SuperCIFAR100(root='../data', train=True, download=False, transform=tf)
dataloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

# 设置优化器和损失函数
def train_model(model, dataloader, epochs, device, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Continued Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

# 加载已训练的模型
model_path = f'./models40epoch/resnet50/resnet50_gen{args.gennum}.pth' 
model = load_model(model_path, 20, device)  
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 继续训练模型
train_model(model, dataloader, 10, device, optimizer, criterion)  # 这里设置额外训练的周期数为10

# 保存更新后的模型
os.makedirs('./models/resnet50/', exist_ok=True)
torch.save(model.state_dict(), f'./models/resnet50/resnet50_updated_gen{args.gennum}.pth')
print("Updated model saved.")
