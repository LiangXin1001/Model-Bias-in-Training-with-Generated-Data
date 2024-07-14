import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid,save_image
import wandb
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils import utils


wandb.login(key="fee5399c08df3ebfcc0cdbc4755392c5011a8133")
# 设置固定的种子以确保结果可重复
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 初始化 wandb
wandb.init(
    project="DCGAN-project_turing",
    config={
        "learning_rate_d": 1e-4,    
        "learning_rate_g": 1e-4,
        "epochs": 50,
        "batch_size": 32,
        "n_critic": 5,
        "n_generator":5,
    }
)



# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Train a Conditional GAN with MNIST')
 
parser.add_argument('--base_dir', type=str, required=True, help='Base directory for saving csv files and images')
parser.add_argument('--images_dir', type=str, required=True, help='Base directory for saving csv files and images')
parser.add_argument('--model_dir', type=str, required=True, help='Base directory for saving models ')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
parser.add_argument('--n_critic', type=int, default=5, help='Number of training steps for the critic per epoch')
parser.add_argument('--n_generator', type=int, default=5, help='Number of training steps for the generator per epoch')
parser.add_argument('--display_step', type=int, default=100, help='Interval of epochs to display training progress')
parser.add_argument('--genmodel', type=str, default=100, help='Interval of epochs to display training progress')


args = parser.parse_args()
 

# # 配置文件和图像路径
# csv_file = '/local/scratch/hcui25/Project/xin/CS/GAN/datasets/top_10_percent_images.csv'  # 更新CSV文件路径
# image_folder = '/local/scratch/hcui25/Project/xin/CS/GAN/datasets/Colored_MNIST_Bais'  # 更新图像文件夹路径
# 路径设置 
# base_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation'
# train_csv = os.path.join(base_dir, 'train_labels.csv') 
train_csv = os.path.join(args.base_dir, 'train.csv') 
train_img_dir = args.images_dir

test_csv = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv'
test_img_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test'
 

mean_rgb , std_rgb = utils.get_mean_std(train_csv, train_img_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = mean_rgb, std = std_rgb)  # 对三个通道使用相同的均值和标准差
 
])

import os
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径,包含图片名称和标签。
            img_dir (string): 包含所有图片的目录路径。
            transform (callable, optional): 可选的转换函数，应用于样本。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')   
        label = self.img_labels.iloc[idx, 1]
        color = self.img_labels.iloc[idx, 2]  # 加载颜色标签
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)  # 默认转换为张量
        return image, label, color  # 返回图像、数值标签和颜色标签




# 创建数据集实例
train_dataset = CustomMNISTDataset(csv_file=train_csv, img_dir=train_img_dir, transform=ToTensor())
test_dataset = CustomMNISTDataset(csv_file=test_csv, img_dir=test_img_dir, transform=ToTensor())

batch_size = 32
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        self.color_emb = nn.Embedding(3, 10)  # 假设有4种颜色

        self.model = nn.Sequential(
            nn.Linear(2352 + 10 +10, 1024),  # 调整输入大小为图像大小加上标签和颜色嵌入
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels,colors):
        x = x.view(x.size(0), -1)  # 改为自动计算输入张量的维度
        c = self.label_emb(labels)
        color_c = self.color_emb(colors)
        x = torch.cat([x, c, color_c], 1)
       
        out = self.model(x)
       
        return out.squeeze()
 

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        self.color_emb = nn.Embedding(3,10)
        
        self.model = nn.Sequential(
            nn.Linear(110 +10, 256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),  # 添加BatchNorm层
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),  # 添加BatchNorm层
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),  # 添加BatchNorm层
            nn.Linear(1024, 2352),  # 更新此处以适应彩色图像
            nn.Tanh()  # 保持 Tanh 作为最后一层，确保输出值在 [-1, 1] 范围内
        )
    
    def forward(self, z, labels,colors):
        z = z.view(z.size(0), 100)  # 确保输入向量正确展开
        c = self.label_emb(labels)
        color_c = self.color_emb(colors)
        x = torch.cat([z, c, color_c], 1)    
         
        # # 调试信息
        # print(f"z shape: {z.shape}")
        # print(f"label embedding shape: {c.shape}")
        # print(f"color embedding shape: {color_c.shape}")

        out = self.model(x)      
        return out.view(x.size(0), 3, 28, 28)  # 调整输出维度以匹配三通道彩色图像


generator = Generator().cuda()
discriminator = Discriminator().cuda()


criterion = nn.BCELoss()
# 调整优化器的学习率
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)   
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)   




def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    # 根据实际颜色标签数量随机生成颜色标签
    fake_colors = Variable(torch.LongTensor(np.random.randint(0, 3, batch_size))).cuda()  # 颜色标签是0, 1, 2
    fake_images = generator(z, fake_labels,fake_colors)
    validity = discriminator(fake_images, fake_labels,fake_colors)
    gan_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    
    # 计算前景和背景掩码
    foreground_mask, background_mask = calculate_foreground_background_mask(fake_images[0])
    # 计算背景损失
    bg_loss = background_loss(fake_images, background_mask)
    # 计算颜色损失
    col_loss = color_loss(fake_images, foreground_mask)

    
    #总损失 + 1 * bg_loss + 1 * col_loss
    g_loss =  gan_loss  
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels,colors):
    d_optimizer.zero_grad()
    # # 使用标签平滑，真实标签从1改为0.9
    # real_labels = Variable(torch.ones(batch_size) * 0.9).cuda()
    fake_labels = Variable(torch.zeros(batch_size)).cuda()
   

    # train with real images
    real_validity = discriminator(real_images, labels,colors)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size) ).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels_input = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_colors = Variable(torch.LongTensor(np.random.randint(0, 3, batch_size))).cuda()  # 同上，颜色标签是0, 1, 2
    fake_images = generator(z, fake_labels_input,fake_colors)
    
    fake_validity = discriminator(fake_images, fake_labels_input,fake_colors)
    fake_loss = criterion(fake_validity, fake_labels)
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    d_optimizer.step()
    return d_loss.item()

#加颜色loss

#计算图像的前景（foreground）和背景（background）掩码
def calculate_foreground_background_mask(image, threshold=0.05):
    # 将图像转换为灰度来计算亮度
    gray_scale = image.mean(dim=0)  # 假设图像为 [C, H, W], 平均C通道
    # 创建前景掩码，亮度大于阈值的为前景
    foreground_mask = (gray_scale > threshold).float()
    # 背景掩码是前景掩码的逆
    background_mask = 1 - foreground_mask
    return foreground_mask, background_mask

# 定义背景损失函数，惩罚生成图像中非数字区域不是黑色的部分
def background_loss(fake_images, background_mask):
    return torch.mean((fake_images * background_mask) ** 2)


# 定义颜色损失函数，限制数字颜色为红、蓝、绿三种颜色
def color_loss(fake_images, foreground_mask):
    # 假设生成的图像为 [B, C, H, W]
    red_channel = fake_images[:, 0, :, :]
    green_channel = fake_images[:, 1, :, :]
    blue_channel = fake_images[:, 2, :, :]
    
    # 计算每个像素的颜色距离到红、蓝、绿三种颜色的距离
    red_distance = (red_channel - 1) ** 2 + green_channel ** 2 + blue_channel ** 2
    green_distance = red_channel ** 2 + (green_channel - 1) ** 2 + blue_channel ** 2
    blue_distance = red_channel ** 2 + green_channel ** 2 + (blue_channel - 1) ** 2
    
    # 找到最小距离
    min_distance = torch.min(torch.min(red_distance, green_distance), blue_distance)
    
    return torch.mean(min_distance * foreground_mask)




 
torch.cuda.empty_cache()


for epoch in range(args.num_epochs):
    print('Starting epoch {}...'.format(epoch), end=' ')
    for i, (images, labels, colors) in enumerate(train_loader):
        
        step = epoch * len(train_loader) + i + 1
        real_images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        colors = Variable(colors).cuda()  # 确保颜色标签也传给模型
        generator.train()
        
        d_loss = 0
        for _ in range(args.n_critic):
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                              generator, d_optimizer, criterion,
                                              real_images, labels,colors)
        # # 生成器训练，增加训练次数
        for _ in range(args.n_generator):
            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
            print(f'Step {step}: Generator Loss: {g_loss}, Discriminator Loss: {d_loss}')
            wandb.log({"Generator Loss": g_loss, "Discriminator Loss": d_loss / args.n_critic, "Epoch": epoch, "Step": step})
        # g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)

        # # 使用 WandB 记录损失
        # wandb.log({"Generator Loss": g_loss, "Discriminator Loss": d_loss / args.n_critic, "Epoch": epoch, "Step": step})

        # g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
        
        print(f'i {i} , Step {step}: Generator Loss: {g_loss}, Discriminator Loss: {d_loss / args.n_critic}')
        if step % args.display_step == 0:
            print("hello")
            generator.eval()
            # z = Variable(torch.randn(9, 100)).cuda()
            # labels = Variable(torch.LongTensor(np.arange(9))).cuda()
            z = torch.randn(9, 100).cuda()
            labels = torch.LongTensor(np.arange(9)).cuda()
            colors = torch.LongTensor(np.random.randint(0, 3, 9)).cuda()
            sample_images = generator(z, labels,colors)
            # sample_images = generator(z, labels).unsqueeze(1)
            grid = make_grid(sample_images, nrow=3, normalize=True)
            save_path = f'/local/scratch/hcui25/Project/xin/CS/GAN/datasets/CGANtmpimages3470/step_{step}.png'
            save_image(grid, save_path)
            print(f'Saved sample images to {save_path}')
        
            # 记录生成的图片到 WandB
            # wandb.log({"Generated Images": [wandb.Image(grid, caption=f"Step {step}")]})
        
    # g_scheduler.step()
    # d_scheduler.step()   
    print('Done!')

# model_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model'
# 再次保存模型和优化器状态
torch.save(generator.state_dict(), os.path.join(args.model_dir, args.genmodel))
torch.save(d_optimizer.state_dict(), os.path.join(args.model_dir, 'g_optimizer_state1gen50epoch123.pt'))
torch.save(discriminator.state_dict(), os.path.join(args.model_dir, 'discriminator_state1gen50epoch123.pt'))
torch.save(d_optimizer.state_dict(), os.path.join(args.model_dir, 'd_optimizer_state1gen50epoch123.pt'))

z = Variable(torch.randn(100, 100)).cuda()
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
 