import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
sys.path.append('../')  # 将上一级目录添加到系统路径
from datasets import SuperCIFAR100, GeneratedDataset, tf 
from torch.utils.data import ConcatDataset, DataLoader 

 
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier with custom dataset')
    parser.add_argument('--gennum', type=int, required=True, help='Generator number for filename customization')
    parser.add_argument('--data_root_paths', type=str, required=True, help='Directory path to save models')
    parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_arguments()
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
        model = inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model.to(device)

trainset = SuperCIFAR100(root='../data', train=True, download=False, transform=tf)
    
#prepare datasets
if args.data_root_paths:
    generated_dataset = GeneratedDataset(root_dir=args.data_root_paths, transform=tf)

    combined_dataset = ConcatDataset([generated_dataset, trainset])

    # 为训练创建 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4 
    )
else:
    dataloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=64,
    shuffle=True,
    num_workers=4 
    )


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
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
# 保存模型函数
def save_model(model, model_name, epoch, path):
    os.makedirs(path, exist_ok=True)
    model_filename = f"{model_name}_gen{args.gennum}.pth"
    torch.save(model.state_dict(), os.path.join(path, model_filename))
    print(f"Model saved: {model_filename}")

# Instantiate and configure the model
model = get_model(args.model_name, 20, device)  # Adjust num_classes based on your needs
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
train_model(model, dataloader, args.epochs, device, optimizer, criterion)
path = f"./models/{model_name}"
os.makedirs(path, exist_ok=True)
save_model(model, model_name, epoch, path)