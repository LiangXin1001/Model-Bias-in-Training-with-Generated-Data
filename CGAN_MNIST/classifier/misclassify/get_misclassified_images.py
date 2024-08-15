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
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
 
from torchvision import transforms
from torch.utils.data import DataLoader
import seaborn as sns
import argparse
 
import matplotlib.pyplot as plt
 

import os
import pandas as pd
import numpy as np
from PIL import Image

def get_mean_std(csv_file, image_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 初始化总和和平方总和变量，以及图像计数器
    sum_rgb = np.zeros(3)
    sum_squares_rgb = np.zeros(3)
    image_count = 0

    # 遍历图像路径，计算均值和方差
    for image_name in df['image_name']:
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0  # 转换为0-1范围
                sum_rgb += img_array.sum(axis=(0, 1))  # 对所有像素进行加和
                sum_squares_rgb += (img_array**2).sum(axis=(0, 1))  # 对所有像素的平方进行加和
            image_count += img_array.shape[0] * img_array.shape[1]  # 总像素数
        else:
            print(f"Image {image_name} not found in the directory.")

    if image_count == 0:
        raise ValueError("No images found in the specified directory.")

    # 计算全局均值
    mean_rgb = sum_rgb / image_count

    # 计算全局标准差
    # std_rgb = np.sqrt(sum_squares_rgb / image_count - mean_rgb**2)
    # 在计算标准差之前增加小的正数
    std_rgb = np.sqrt(sum_squares_rgb / image_count - mean_rgb**2 + 1e-8)

    print(f"Mean: {mean_rgb}")
    print(f"Std: {std_rgb}")
    
    return mean_rgb, std_rgb


# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--test_csv', type=str, required=True, help='CSV file path for testing dataset')
 
parser.add_argument('--test_images_dir', type=str, required=True, help='Directory path for testing images')
parser.add_argument('--model_save_path', type=str, required=True, help='Directory path to save models')
parser.add_argument('--result_save_path', type=str, required=True, help='Directory path to save results')
parser.add_argument('--model_name', type=str, required=True, help='name  models')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--gennum', type=int, required=True,  help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

args = parser.parse_args()

if not os.path.exists(args.result_save_path):
    os.makedirs(args.result_save_path)

 
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[idx, 1]
        color = self.annotations.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, label, color, img_path   

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def test(model, test_loader, misclassified_dir):
    if not os.path.exists(misclassified_dir):
        os.makedirs(misclassified_dir)  

    correct = 0
    misclassified_records = [] 
    with torch.no_grad():
        for data, target, color, paths in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
 
            for i in range(data.size(0)):
                if pred[i] != target[i]:
                    print( os.path.basename(paths[i]), " True Label ",int(target[i].item())," Predicted Label ",int(pred[i].item()))
                    shutil.copy(paths[i], os.path.join(misclassified_dir, os.path.basename(paths[i]))) 
                    misclassified_records.append({
                        'Filename': os.path.basename(paths[i]),
                        'True Label': int(target[i].item()),
                        'Predicted Label': int(pred[i].item())
                    }) 

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test accuracy: {accuracy}%')
    df_misclassified = pd.DataFrame(misclassified_records)
    df_misclassified.to_csv(os.path.join(args.result_save_path, 'misclassified_images.csv'), index=False)
    print("Misclassified images data saved to CSV.")





def main():
    model_path = os.path.join(args.model_save_path, f"{args.model_name}_epoch{args.epochs}.pth")
    model = get_model(args.model_name, 10, device)
    mean_rgb , std_rgb =  get_mean_std(args.test_csv, args.test_images_dir)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean = mean_rgb, std = std_rgb)  
    ])
    test_dataset = CustomDataset(csv_file=args.test_csv, root_dir=args.test_images_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    misclassified_dir = os.path.join(args.result_save_path, 'misclassified_images')   
    test(model, test_loader, misclassified_dir)

if __name__ == '__main__':
    main()
