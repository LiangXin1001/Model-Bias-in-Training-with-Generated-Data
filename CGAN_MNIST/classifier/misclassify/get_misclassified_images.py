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



# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--test_csv', type=str, required=True, help='CSV file path for testing dataset')
 
parser.add_argument('--test_images_dir', type=str, required=True, help='Directory path for testing images')
parser.add_argument('--model_save_path', type=str, required=True, help='Directory path to save models')
parser.add_argument('--result_save_path', type=str, required=True, help='Directory path to save results')
parser.add_argument('--model_name', type=str, required=True, help='name  models')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
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

def load_model(model_path):
    # 创建一个与训练相同的模型实例，这里使用 ResNet50
    model = resnet50(pretrained=False)
    num_features = model.fc.in_features
    # 如果模型的最后一层在训练中被修改了，确保这里也要做相应的修改
    model.fc = nn.Linear(num_features, 10)  #  模型最后有10个输出

    # 加载保存的权重
    model.load_state_dict(torch.load(model_path))

    # 将模型设置为评估模式
    model.eval()

    return model
 

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
    model = load_model(model_path)
    model = model.to(device)
  
    mean_rgb , std_rgb = utils.get_mean_std(args.test_csv, args.test_images_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb)
    ])
    test_dataset = CustomDataset(csv_file=args.test_csv, root_dir=args.test_images_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    misclassified_dir = os.path.join(args.result_save_path, 'misclassified_images')   
    test(model, test_loader, misclassified_dir)

if __name__ == '__main__':
    main()
