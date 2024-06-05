import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
      

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x


def get_dataloader(dataset_name, train=True):
 

        # 使用 transforms 将所有图像转为灰度图像
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小以适应模型
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])


    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        data = dataset(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform.transforms.insert(0, transforms.Grayscale(num_output_channels=1))
        dataset = torchvision.datasets.CIFAR10
        data = dataset(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        data = dataset(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        transform.transforms.insert(0, transforms.Grayscale(num_output_channels=1))
        dataset = torchvision.datasets.SVHN
        data = dataset(root='./data', split='train' if train else 'test',
                download=True, transform=transform)

    else:
        # 如果没有匹配的数据集名称，返回None或抛出异常
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    dataloader.dataset.dataset_name = dataset_name  # 给dataloader添加属性以便打印
    return dataloader




def finetune_and_evaluate(model, train_loader, test_loader, epochs, optimizer, criterion):
    # 微调模型
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on {test_loader.dataset.dataset_name}: {accuracy}%')
    return accuracy




import gc
import torch 
torch.cuda.empty_cache()  # 清空CUDA缓存
gc.collect()  # 垃圾回收

# 主程序
def main():
    # 'MNIST', 'CIFAR10', 'FashionMNIST',
    datasets = [ 'SVHN']
    results = {}

    for dataset_name in datasets:
        print(f'Starting training and testing on {dataset_name}')
        model = SimpleCNN(num_classes=10 ).to(device)
        train_loader = get_dataloader(dataset_name, train=True)
        test_loader = get_dataloader(dataset_name, train=False)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 微调和评估
        accuracy = finetune_and_evaluate(model, train_loader, test_loader, 5, optimizer, criterion)

        # accuracy = finetune_and_evaluate(model, train_loader, test_loader, epochs=5, optimizer, criterion)
        results[dataset_name] = accuracy

    # 打印所有结果
    print("Final results across datasets:")
    for dataset_name, accuracy in results.items():
        print(f'{dataset_name}: {accuracy}%')

if __name__ == '__main__':
    main()