import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import os 
import pandas as pd
import torch.nn as nn
import argparse
import csv
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
 
 
 

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Train a Conditional GAN with MNIST')
 
parser.add_argument('--images_dir', type=str, required=True, help='Base directory for saving generated images')
parser.add_argument('--model_dir', type=str, required=True, help='Base directory for saving models ')
parser.add_argument('--genmodel', type=str, default=100, help='Interval of epochs to display training progress')
parser.add_argument('--gen_num', type=int, default=1, help='Interval of epochs to display training progress')

parser.add_argument('--traincsv', type=str, default=100, help='Csv file for trainset')
parser.add_argument('--synthetizedtraincsv', type=str, default=100, help='Csv file for trainset')
 
args = parser.parse_args()
 
  
# 创建目录
 
os.makedirs(args.images_dir, exist_ok=True)
 # args.synthetizedtraincsv包含了CSV文件的路径
csv_path = args.synthetizedtraincsv
csv_dir = os.path.dirname(csv_path)  # 获取文件路径中的目录部分

# 确保目录存在
os.makedirs(csv_dir, exist_ok=True)
 
data = pd.read_csv(args.traincsv)


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



# 加载模型
generator = Generator().cuda()
generator.load_state_dict(torch.load(os.path.join(args.model_dir, args.genmodel)))
generator.eval()
 
# 计算每个label和颜色的组合数量
counts = data.groupby(['label', 'color']).size().reset_index(name='count')

#初始化csv文件

with open(args.synthetizedtraincsv, 'x', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['image_name','label','color'])
    # 生成图像
    for _,row in counts.iterrows():
        num_images = row['count']
        label = row['label']
        color = row['color']
        z = torch.randn(num_images, 100).cuda() 
        labels = torch.LongTensor([label] * num_images).cuda()  # 为每个数字生成10个相同的标签
        colors = torch.LongTensor([color] * num_images).cuda()  # 随机颜色标签
        images = generator(z, labels, colors)
        folder_path = args.images_dir
        os.makedirs(folder_path, exist_ok=True)
        for i , img in enumerate(images):
            save_image(img,os.path.join(folder_path,f'image_{label}_{color}_{i}_gen{args.gen_num}.png'),normalize = True)
            csvwriter.writerow([f'image_{label}_{color}_{i}_gen{args.gen_num}.png', label, color])


print("generate image , done.")
