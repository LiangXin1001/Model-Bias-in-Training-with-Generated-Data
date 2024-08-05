from PIL import Image
import numpy as np
import pandas as pd
import os


 
 

def get_mean_std(csv_file, image_dirs):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 初始化总和和平方总和变量，以及图像计数器
    sum_rgb = np.zeros(3)
    sum_squares_rgb = np.zeros(3)
    image_count = 0

    # 遍历图像路径，计算均值和方差
    for image_name in df['image_name']:
        image_found = False
        for image_dir in image_dirs:
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img_array = np.array(img) / 255.0  # 转换为0-1范围
                    sum_rgb += img_array.sum(axis=(0, 1))  # 对所有像素进行加和
                    sum_squares_rgb += (img_array**2).sum(axis=(0, 1))  # 对所有像素的平方进行加和
                image_count += img_array.shape[0] * img_array.shape[1]  # 总像素数
                image_found = True
                break
        if not image_found:
            print(f"Image {image_name} not found in any of the directories.")

    if image_count == 0:
        raise ValueError("No images found in the specified directories.")

    # 计算全局均值
    mean_rgb = sum_rgb / image_count

    # 计算全局标准差
    std_rgb = np.sqrt(sum_squares_rgb / image_count - mean_rgb**2)
    mean_rgb = np.array(mean_rgb)
    std_rgb = np.array(std_rgb)
    print(f"Mean: {mean_rgb}")
    print(f"Std: {std_rgb}")
    return mean_rgb, std_rgb
