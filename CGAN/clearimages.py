import os
import shutil
import csv
import argparse


# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Train a Conditional GAN with MNIST')
 
parser.add_argument('--folder_a', type=str, required=True, help='Base directory for saving generated images')
parser.add_argument('--folder_b', type=str, required=True, help='Base directory for saving models ')
parser.add_argument('--csv_file', type=str, default=100, help='Interval of epochs to display training progress')
 
args = parser.parse_args()



# 设置文件夹和CSV文件路径
folder_a =  args.folder_a
folder_b =  args.folder_b
csv_file =  args.csv_file

# 从CSV文件中读取图片名
with open(csv_file, newline='') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    images_from_csv = {row[0] for row in reader}

# 获取文件夹A和B中的所有图片文件
images_in_a = set(os.listdir(folder_a))
images_in_b = set(os.listdir(folder_b))

# 复制文件夹B中的图片到A，如果它们在CSV中但不在A中
for image in images_from_csv:
    if image in images_in_b and image not in images_in_a:
        shutil.copy(os.path.join(folder_b, image), os.path.join(folder_a, image))
 

print("操作完成。")