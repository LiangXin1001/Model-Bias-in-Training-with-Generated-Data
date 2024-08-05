import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
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
parser.add_argument('--model_name', type=str, required=True, help='name  models')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')

args = parser.parse_args()

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_weight_name = 'resnet50'
# 配置文件和图像路径
csv_file =  args.train_csv
image_folder = args.train_images_dir.split(',')

mean_rgb , std_rgb = utils.get_mean_std(csv_file, image_folder)

# 数据转换，适用于彩色图像
transform = transforms.Compose([
    
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
      transforms.Normalize(mean = mean_rgb, std = std_rgb)  
])

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

 
model = resnet50(weights=False).to(device)

num_features = model.fc.in_features  # 获取最后一个线性层的输入特征数量
model.fc = nn.Linear(num_features, 10).to(device)  # 替换为新的线性层，适配10类

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
                

# 测试函数
def test():
    # model_path = os.path.join(args.model_save_path, f"{args.model_name}_epoch{args.epochs}.pth")
 
    # model = resnet50(weights=None)  # 如果你修改了模型，请确保这里也做相应的修改
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)  # 假设你的模型输出10个类别
    # model.load_state_dict(torch.load(model_path))
    # model = model.to(device)
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_colors = []
    with torch.no_grad():
        for data, target, color in test_loader:
            data, target,color = data.to(device), target.to(device),color.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失加和
            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最高的索引
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_colors.extend(color.cpu().numpy()) #收集颜色信息

    # 计算混淆矩阵及其它统计指标
    cm = confusion_matrix(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    
   
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    print('Confusion Matrix:\n', cm)





    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_colors = np.array(all_colors)

    
    # Difference in Means
    correct_main = np.sum([all_preds[i] == all_targets[i] for i in indices_main])
    correct_not_main = np.sum([all_preds[i] == all_targets[i] for i in indices_not_main])
    mean_diff = correct_main / len(indices_main) - correct_not_main / len(indices_not_main)
    print(f'Difference in Means: {mean_diff}')

    # 计算 TPR FPR 
    num_classes = 10
    num_colors =3
    
    # 初始化TPR和FPR字典
    tpr = {}
    fpr = {}

    # 对每个数字计算TPR和FPR
    for digit in range(num_classes):
        tpr[digit] = np.zeros(num_colors)
        fpr[digit] = np.zeros(num_colors)

        for color in range(num_colors):
            # 筛选出属于当前数字和颜色的样本
            indices = (all_targets == digit) & (all_colors == color)
            true_indices = all_targets == digit
            pred_indices = all_preds == digit

            TP = np.sum(pred_indices & true_indices & indices)
            FN = np.sum(~pred_indices & true_indices & indices)
            FP = np.sum(pred_indices & ~true_indices & indices)
            TN = np.sum(~pred_indices & ~true_indices & indices)

            # 计算TPR和FPR
            if TP + FN > 0:
                tpr[digit][color] = TP / (TP + FN)
            if TN + FP > 0:
                fpr[digit][color] = FP / (TN + FP)

    txt_file_path = os.path.join(args.result_save_path,'tpr_fpr_metrics.txt')
    
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    with open(txt_file_path, 'w') as file:
        file.write('Digit,Color,TPR,FPR\n')
        for digit in tpr:
            for color in range(len(tpr[digit])):
                file.write(f"{digit},{color},{tpr[digit][color]:.2f},{fpr[digit][color]:.2f}\n")

    print("Data saved to tpr_fpr_metrics.txt")

    # 为每种颜色生成和保存混淆矩阵
    color_map = {0: 'Red', 1: 'Blue', 2: 'Green'}
      # 创建结果保存目录
    result_save_path =  args.result_save_path
    os.makedirs(result_save_path, exist_ok=True)


    for color, color_name in color_map.items():
        mask = (all_colors == color)
        preds_color = all_preds[mask]
        targets_color = all_targets[mask]
        cm = confusion_matrix(targets_color, preds_color)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {color_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(result_save_path, f'confusion_matrix_{color_name}_{model_weight_name}.png'))
        plt.close()

    print("All confusion matrices saved.")



    # 计算每个数字在每种颜色下的分类成功率
    unique_labels = np.unique(all_targets)
    unique_colors = np.unique(all_colors)

   
    # 存储结果的DataFrame
    results = []

    for label in unique_labels:
        for color in unique_colors:
            # 过滤出当前数字和颜色的样本
            mask = (all_targets == label) & (all_colors == color)
            if np.sum(mask) == 0:
                continue
            true_positive = np.sum((all_preds[mask] == label))
        
            accuracy = true_positive / np.sum(mask)
            results.append({"Digit": label, "Color": color, "Accuracy": accuracy})

    results_df = pd.DataFrame(results)

    # 保存结果到CSV文件
    results_csv_path = os.path.join(result_save_path, f'classification_accuracy_by_color_82{model_weight_name}.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # 可视化结果并保存图像
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        subset = results_df[results_df["Digit"] == label]
        plt.plot(subset["Color"], subset["Accuracy"], marker='o', label=f"Digit {label}")

    plt.xlabel("Color")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy for Each Digit by Color")
    plt.legend()
    plt.grid(True)

    # 保存图像到文件
    accuracy_plot_path = os.path.join(result_save_path, f'classification_accuracy_by_color_{model_weight_name}.png')
    plt.savefig(accuracy_plot_path)
    print(f"Plot saved to {accuracy_plot_path}")
    plt.close()  # 关闭绘图以释放内存

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)

    def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(result_save_path, f'confusion_matrix_{model_weight_name}.png'))
        plt.close()

    labels = [str(i) for i in range(10)]  # 假设标签为0到9
    plot_confusion_matrix(cm, labels)
















# 保存模型函数
def save_model(model, model_name,  epochs, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f"{model_name}_epoch{args.epochs}.pth"))
  
# 训练和测试模型
for epoch in range(1, args.epochs + 1):
    train(epoch)
     
    save_model(model, args.model_name, args.epochs, args.model_save_path)


