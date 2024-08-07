from PIL import Image
import pandas as pd
import os
from random import randint

# 假设 CSV 文件路径和图片目录
csv_path = 'augment/adjusted_train_dataset.csv'
df = pd.read_csv(csv_path)
df['color'] = df['color'].astype(int)
base_image_dir = 'MNIST/mnist_train'
 
min_color_count = 2000
 
label_color_counts = df.groupby(['label', 'color']).size().reset_index(name='count')
label_color_counts['needed'] = label_color_counts['count'].apply(lambda x: max(min_color_count - x, 0))

# 图片旋转函数，保持28x28尺寸
def rotate_image(image_path, output_path, angle):
    img = Image.open(image_path)
    # 旋转图像，填充黑色
    rotated_img = img.rotate(angle, fillcolor='black', expand=False)
    # 如果需要保持图像尺寸为28x28，可以使用crop或resize确保尺寸
    rotated_img = rotated_img.resize((28, 28), Image.Resampling.LANCZOS)
    rotated_img.save(output_path)

# 创建增强图片
new_images = []
for index, row in label_color_counts.iterrows():
    if row['needed'] > 0:
         
        sample_images = df[(df['label'] == row['label']) & (df['color'] == row['color'])].sample(n=row['needed'], replace=True, random_state=42)
        for idx, image_row in sample_images.iterrows():
            angle = randint(-15, 15)   
            new_name = image_row['image_name'].replace('.png', f'_rotated_{angle}.png')
            new_path =  os.path.join(base_image_dir , new_name)
            original_image_path = os.path.join(base_image_dir, image_row['image_name'])
            rotate_image(original_image_path, new_path, angle)
            new_images.append({'image_name': new_name, 'label': row['label'], 'color': row['color']})
            print("new_images.append new_name:",new_name)
 
additional_images_df = pd.DataFrame(new_images)
combined_df = pd.concat([df, additional_images_df], ignore_index=True)

new_csv_path = os.path.join('augment', 'augmented_train_dataset.csv')
combined_df.to_csv(new_csv_path, index=False)