import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.transforms as transforms
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
from torch.utils.data import ConcatDataset, DataLoader 
import os
 
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier with custom dataset')
    parser.add_argument('--gennum', type=int, required=True, help='Generator number for filename customization')
    parser.add_argument('--data_root_paths', type=str,  default= "",help='Directory path to save models')
    parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train the model')
    parser.add_argument('--start_train_epoch', type=int, default=0, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_arguments()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
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
    elif model_name.lower() == 'simplecnn':
        model = SimpleCNN(num_classes)
    elif model_name.lower() == 'mobilenetv3':
        model = mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name.lower() == 'inceptionv4':
        model = inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model.to(device)

  
trainset = SuperCIFAR100(root='../data', train=True, download=False, transform=tf)
    
#prepare datasets
if args.data_root_paths:
    root_dirs = args.data_root_paths.split(',') 
    generated_dataset = GeneratedDataset(root_dirs=root_dirs, transform=tf)

    combined_dataset = ConcatDataset([generated_dataset, trainset])

    # 为训练创建 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4 
    )
else:
    print("gen 0 , use SuperCIFAR100 trainset")
    dataloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=64,
    shuffle=True,
    num_workers=4 
    )


def train_model(model, dataloader, start_epoch, epochs, device, optimizer, criterion, model_name):
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        for batch_idx, (data, target, _) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        save_path = f"./models/{args.model_name}/gen{args.gennum}" 
        if epoch % 5 == 0:  # 每5个epoch保存一次

            save_model(model, optimizer, model_name, epoch,save_path) 

def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded model and optimizer from {load_path}")

def save_model(model, optimizer, model_name, epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"{model_name}_epoch_{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
    print(f"Model and optimizer states saved to {model_path}")



# Instantiate and configure the model
model = get_model(args.model_name, 20, device)   
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.start_train_epoch > 1:
    load_path = f"./models/{args.model_name}/gen{args.gennum}/{args.model_name}_epoch_{args.start_train_epoch - 1}.pth"
    load_model(model, optimizer, load_path)

# Train the model starting from the specified epoch
train_model(model, dataloader, args.start_train_epoch, args.epochs, device, optimizer, criterion, args.model_name)
 
path = f"./models/{args.model_name}/gen{args.gennum}"
os.makedirs(path, exist_ok=True)
save_model(model,optimizer,args.model_name, args.epochs, path)
 