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
# 加载模型
def load_model(model_path):
    model = resnet50(weights=None)  # 如果你修改了模型，请确保这里也做相应的修改
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # 假设你的模型输出10个类别
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 测试函数
def test(model, test_loader):
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_colors = []
    with torch.no_grad():
        for data, target, color in test_loader:
            data, target,color = data.to(device), target.to(device),color.to(device)
            output = model(data)
           
            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最高的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_colors.extend(color.cpu().numpy()) #收集颜色信息
 
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 创建一个 DataFrame
    df = pd.DataFrame({
        'Predictions': all_preds,
        'Targets': all_targets,
        'Colors': all_colors
    })
    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_colors = np.array(all_colors)

    # 定义颜色和数字映射
    color_map = {0: 'Red', 1: 'Blue', 2: 'Green'}
    num_classes = 10  

    # 存储结果
    results = []

    # 计算每个数字和颜色组合的正确率
    for label in range(num_classes):
        for color in range(3):
            subset = df[(df['Targets'] == label) & (df['Colors'] == color)]
            accuracy = np.mean(subset['Predictions'] == subset['Targets'])
            results.append({'Digit': label, 'Color': color_map[color], 'Accuracy': accuracy})

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    print("results : ")
    print(results)
    print(results_df.head())  # 打印前几行数据
    print(results_df.describe())  # 获取数据的描述性统计信息

    # 绘制图表
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Digit', y='Accuracy', hue='Color', data=results_df)
    plt.title('Classification Accuracy for Each Digit by Color')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.legend(title='Color')
    plt.savefig(os.path.join(args.result_save_path, f'Classification_Accuracyepoch.png'))
     

    # 按照数字分组，然后计算每组的标准差
    std_devs = results_df.groupby('Digit')['Accuracy'].std()
    print("std_devs : ",std_devs)
    std_devs.plot(kind='bar', color='skyblue', figsize=(10, 6))
    plt.title('Standard Deviation of Accuracy for Each Digit')
    plt.xlabel('Digit')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(args.result_save_path, 'std_devs_bar_chart.png'))  # 保存图像
    
    
    # 对每个数字分组，然后找出每组中准确率最高的记录
    highest_accuracy_per_digit = results_df.loc[results_df.groupby('Digit')['Accuracy'].idxmax()]
 
    threshold = 0.1

    # 根据标准差的阈值调整颜色
    colors = []
    for idx, row in highest_accuracy_per_digit.iterrows():
        digit = row['Digit']
        if np.isclose(std_devs.loc[digit], 0.0, atol=1e-8):
            colors.append('gray')
        elif std_devs.loc[digit] > threshold:
            colors.append(row['Color'])
        else:
            colors.append('yellow')  # 标准差不超过阈值的数字使用黄色
    # 打印结果
    print(highest_accuracy_per_digit)
    plt.figure(figsize=(10, 6))
    plt.scatter(highest_accuracy_per_digit['Digit'], [1]*len(highest_accuracy_per_digit), color=colors, s=1000, edgecolor='black')
    plt.yticks([])
    plt.title('Highest Accuracy Color for Each Digit')
    plt.xlabel('Digit')
    plt.xticks(highest_accuracy_per_digit['Digit'])
    plt.grid(True)
    plt.savefig(os.path.join(args.result_save_path, 'highest_accuracy_color.png'))  # 保存图像
    plt.show()
# 主函数
def main():
    model_path = os.path.join(args.model_save_path, f"{args.model_name}_epoch{args.epochs}.pth")
 
    model = load_model(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    mean_rgb , std_rgb = utils.get_mean_std(args.test_csv, [args.test_images_dir])

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean_rgb, std = std_rgb)  # 根据你的设置调整
    ])

    # 加载测试数据集
    test_dataset = CustomDataset(csv_file= args.test_csv, root_dir = args.test_images_dir, transform = transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 进行测试
    test(model, test_loader)

if __name__ == '__main__':
    main()
