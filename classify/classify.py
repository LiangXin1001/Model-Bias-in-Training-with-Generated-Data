import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import wandb
from utils import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse



# 初始化 wandb
wandb.init(
    project="my-ResNet50-project",
    config={
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 8,
    }
)

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
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

args = parser.parse_args()

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_weight_name = 'resnet50'
# 配置文件和图像路径
csv_file =  args.train_csv
image_folder =  args.train_images_dir

mean_rgb , std_rgb = utils.get_mean_std(csv_file, image_folder)

# 数据转换，适用于彩色图像
transform = transforms.Compose([
    
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
      transforms.Normalize(mean = mean_rgb, std = std_rgb)  
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

# 加载数据集
train_csv  = args.train_csv
test_csv =  args.test_csv
dataset_path =  args.train_images_dir
testdatapath =  args.test_images_dir
train_dataset = CustomDataset(csv_file=train_csv, root_dir=dataset_path, transform=transform)
test_dataset = CustomDataset(csv_file=test_csv, root_dir=testdatapath, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# 加载预训练模型
model = resnet50(weights='IMAGENET1K_V1').to(device)

num_features = model.fc.in_features  # 获取最后一个线性层的输入特征数量
model.fc = nn.Linear(num_features, 10).to(device)  # 替换为新的线性层，适配10类

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(epochs):
    model.train()
    for batch_idx, (data, target,_) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Logging to wandb
        wandb.log({"Train Loss": loss.item()})
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {args.epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        torch.cuda.empty_cache()  # 尝试在这里清理CUDA缓存

# 测试函数
def test():
    model.eval()
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
    
    # Logging to wandb
    wandb.log({"Test Loss": test_loss, "Accuracy": accuracy, "Precision": precision, "Recall": recall})
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    print('Confusion Matrix:\n', cm)





    # 转换为NumPy数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_colors = np.array(all_colors)

    color_distribution = {
        0: [0.05, 0.9, 0.05],
        1: [0.05, 0.05, 0.9],
        2: [0.05, 0.9, 0.05],
        3: [0.9, 0.05, 0.05],
        4: [0.05, 0.9, 0.05],
        5: [0.05, 0.05, 0.9],
        6: [0.9, 0.05, 0.05],
        7: [0.05, 0.05, 0.9],
        8: [0.05, 0.9, 0.05],
        9: [0.9, 0.05, 0.05],
    }
    # 计算差异指标
    # 分类主色调和非主色调的索引
    indices_main = [i for i in range(len(all_colors)) if color_distribution[all_targets[i]][all_colors[i]] == 0.9]
    indices_not_main = [i for i in range(len(all_colors)) if color_distribution[all_targets[i]][all_colors[i]] == 0.05]
    
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
    torch.save(model.state_dict(), os.path.join(path, f"{model_name}_epoch{args.epochs}.pth"))
 

# 训练和测试模型
for epoch in range(1, wandb.config.epochs + 1):
    train(args.epochs)
    test()
    save_model(model, args.model_name, args.epochs, args.model_save_path)


