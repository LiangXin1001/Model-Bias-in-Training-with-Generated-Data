import os
import pandas as pd

 
csv_file_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/test.csv'
images_folder_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test'
 

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

# 获取 CSV 中的图片文件名列表
image_files_in_csv = set(df['image_name'].tolist())

# 获取图片文件夹中所有的图片文件名
image_files_in_folder = set(os.listdir(images_folder_path))

# 找出多余的图片文件（存在于文件夹中但不在 CSV 中）
extra_images = image_files_in_folder - image_files_in_csv

# 删除多余的图片文件并记录删除的文件名
deleted_images = []
for extra_image in extra_images:
    image_path = os.path.join(images_folder_path, extra_image)
    if os.path.isfile(image_path):
        os.remove(image_path)
        deleted_images.append(extra_image)

# 输出结果
if deleted_images:
    print(f"以下图片文件存在于文件夹 '{images_folder_path}' 中，但不在 CSV 文件中，并已被删除：")
    for deleted_image in deleted_images:
        print(deleted_image)
else:
    print(f"文件夹 '{images_folder_path}' 中没有多余的图片文件。")

# 可选：输出检查结果到文件
output_file_path = 'deleted_images_report.txt'
with open(output_file_path, 'w') as f:
    if deleted_images:
        f.write(f"以下图片文件存在于文件夹 '{images_folder_path}' 中，但不在 CSV 文件中，并已被删除：\n")
        for deleted_image in deleted_images:
            f.write(f"{deleted_image}\n")
    else:
        f.write(f"文件夹 '{images_folder_path}' 中没有多余的图片文件。\n")

print(f"删除结果已保存到 '{output_file_path}'")