
from PIL import Image, ImageFilter
from collections import Counter
import os
import shutil
import pandas as pd
# # 定义路径
# base_image_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_b160004'
# csv_path = os.path.join(base_image_dir, 'gen1_test_b16.csv')
# target_dir = os.path.join(base_image_dir, 'mix')

# # 确保目标文件夹存在
# os.makedirs(target_dir, exist_ok=True)
# df = pd.read_csv(csv_path)
 

# def detect_and_move_image(image_name):

#     class_folder = image_name.split('_')[0]+'_'+image_name.split('_')[1]
#     image_path = os.path.join(base_image_dir, class_folder, image_name)

#     with Image.open(image_path) as img:
#         img = img.filter(ImageFilter.MedianFilter(size=3))
#         img = img.convert('RGB')
#         pixels = list(img.getdata())
#         color_counts = Counter(pixels)
#         color_counts.pop((0, 0, 0), None)  # 移除黑色背景

#         # 检测红绿蓝颜色的像素数量
#         red_pixels = sum(count for (r, g, b), count in color_counts.items() if r > g + b)
#         green_pixels = sum(count for (r, g, b), count in color_counts.items() if g > r + b)
#         blue_pixels = sum(count for (r, g, b), count in color_counts.items() if b > r + g)

#         # 检测是否为杂色图片
#         total_significant_pixels = red_pixels + green_pixels + blue_pixels
#         if total_significant_pixels > 0:
#             # 检查是否有颜色占比超过50%
#             if (red_pixels < total_significant_pixels * 0.05 or
#                 green_pixels < total_significant_pixels * 0.05 or
#                 blue_pixels < total_significant_pixels * 0.05):
#                 # 移动图片
#                 shutil.move(image_path, os.path.join(target_dir, image_name))
#                 print(f"Moved mixed color image: {image_name}")
#                 return True  # 返回True表示图片被移动
#         return False  # 返回False表示图片未移动


# df['color'] = df['image_name'].apply(detect_and_move_image)

# print("done")


import os
import shutil

# 定义路径
target_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_b160004/mix'
base_image_dir = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_b160004'

# 列出目标文件夹中的所有文件
files = os.listdir(target_dir)

# 将文件移回原始文件夹
for file_name in files:
    # 从文件名中提取类别
    class_folder = file_name.split('_')[0] + '_' + file_name.split('_')[1]
    original_path = os.path.join(base_image_dir, class_folder)
    target_file_path = os.path.join(target_dir, file_name)
    original_file_path = os.path.join(original_path, file_name)
    
    # 检查原始路径是否存在，不存在则创建
    if not os.path.exists(original_path):
        os.makedirs(original_path)

    # 移动文件
    shutil.move(target_file_path, original_file_path)
    print(f"Moved {file_name} back to {original_path}")

print("All files have been moved back to their original folders.")
