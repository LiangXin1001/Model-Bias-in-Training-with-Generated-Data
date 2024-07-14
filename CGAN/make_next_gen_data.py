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

parser.add_argument('--traincsv', type=str, default=100, help='Csv file for trainset')
parser.add_argument('--synthetizedtraincsv', type=str, default=100, help='Csv file for trainset')
parser.add_argument('--top10synthetizedcsv', type=str, default=100, help='Csv file for trainset')
parser.add_argument('--newtraincsv', type=str, default=100, help='Csv file for trainset')
parser.add_argument('--toppercent', type=float, default=0.2, help='Csv file for trainset')
parser.add_argument('--percent_dir', type=str,  required=True,  help=' ')

args = parser.parse_args()
 
  
# 创建目录
os.makedirs(args.percent_dir, exist_ok=True)
os.makedirs(args.images_dir, exist_ok=True)
 
 

#############################################################################
# select for top 20%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载预训练的模型并设置为评估模式
model = resnet18(weights = ResNet18_Weights.DEFAULT)
model = model.to(device)
model.eval()
model.cuda()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_score(image_path):
    """ 评估单个图片的质量分数 """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(image)
        # 使用输出的某种方式来评估质量，这里简单使用最大概率
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score = torch.max(probs).item()
    return score

# 读取文件路径和标签
data = pd.read_csv(args.synthetizedtraincsv)
counts = data.groupby(['label', 'color']).size().reset_index(name='count')
#存储top 20% 信息        
image_data = []
for index, row in counts.iterrows():
    label = row['label']
    color = row['color']
    folder_path = args.images_dir
    images_files = [f for f in os.listdir(folder_path) if f'_{label}_{color}_' in f and f.endswith('.png')]
    images_scores = {file: get_image_score(os.path.join(folder_path,file)) for file in images_files}

    # top 10/20/50/80%
    sorted_images = sorted(images_scores.items(), key=lambda x: x[1],reverse = True)
    top_10_percent = sorted_images[:int(len(sorted_images) * args.toppercent)]

    #保存至单独目录
    # top_images_dir = os.path.join(str(args.toppercent), str(label))
    # os.makedirs(top_images_dir, exist_ok = True)
    for image_file, score in top_10_percent:
        # os.rename(os.path.join(folder_path,image_file),os.path.join(top_images_dir,image_file))
        image_data.append([image_file, label, color]) #添加图片信息

df = pd.DataFrame(image_data, columns = ['image_name', 'label' , 'color'])
df.to_csv(args.top10synthetizedcsv,index = False)

print("Top 10 percent images selection completed.")


################################################################################################


train_df = pd.read_csv(args.traincsv)
top10_df = pd.read_csv(args.top10synthetizedcsv)

updated_train_df = train_df.copy()

for _,row in top10_df.iterrows():
    label = row['label']
    color = row['color']

    # 找到train.csv中第一个匹配的行
    match_index = updated_train_df[(updated_train_df['label'] == label) & (updated_train_df['color'] == color)].index
    if not match_index.empty:  # 确保找到了匹配项
        # 仅删除找到的第一个匹配项
        # print(" delete " , match_index[0])
        updated_train_df = updated_train_df.drop(match_index[0])

# 现在将top10.csv的内容添加到DataFrame中
updated_train_df = pd.concat([updated_train_df, top10_df], ignore_index=True)

# 重置索引，因为删除和添加操作可能会让索引不连续
updated_train_df = updated_train_df.reset_index(drop=True)

# 保存到新的CSV文件
updated_train_df.to_csv(args.newtraincsv, index=False)

print("Updated train.csv with entries from top.csv and saved to updated_train.csv.")
