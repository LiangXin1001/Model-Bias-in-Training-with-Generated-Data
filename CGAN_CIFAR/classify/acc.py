# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.models import resnet50
# import torch.nn as nn
# import torch.optim as optim


# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # 加载训练集和测试集
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
#                                          download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                         download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = resnet50(pretrained=False)  # 不使用预训练权重
# model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 有 100 个类
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# for epoch in range(30):  # 循环多个 epoch
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
        
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         if i % 200 == 199:    # 每 200 mini-batches 打印一次
#             print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
#             running_loss = 0.0
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')























import pandas as pd

def calculate_accuracy(csv_path):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)
    
    # 计算正确预测的数量
    correct_predictions = (data['True Superclass'] == data['Predicted Superclass']).sum()
    
    # 计算总数
    total_predictions = data.shape[0]
    
    # 计算正确率
    accuracy = correct_predictions / total_predictions
    
    return accuracy

# 假设 CSV 文件路径
csv_file_path = 'results/resnet50/test_results_2.csv'  # 更换为实际的文件路径

# 计算并打印正确率
accuracy = calculate_accuracy(csv_file_path)
print(f"Accuracy: {accuracy:.2%}")
