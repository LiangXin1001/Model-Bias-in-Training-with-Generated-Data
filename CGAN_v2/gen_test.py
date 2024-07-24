import os
import torch
import numpy as np
from torchvision import datasets, transforms
import random
from PIL import Image
import csv
import pandas as pd
def colorize_mnist(data, color_distribution, dataset_type):
    # Define colors in RGB
    colors = {
        0: [255, 0, 0],  # Red
        1: [0, 0, 255],  # Blue
        2: [0, 255, 0],  # Green
    }
    
    # Create directories to save images and CSV file
    base_dir = f'./mnist_{dataset_type}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    csv_filename = os.path.join('./', 'train.csv')
    # Prepare to write to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image_name', 'label', 'color'])
        for idx, (image, label) in enumerate(data):
            # Convert image to numpy array
            img_array = np.array(image, dtype=np.float32) / 255

            # Prepare to colorize
            img_color = np.zeros((28, 28, 3), dtype=np.uint8)

            # Randomly decide the color based on the distribution for the given label
            color_choice = random.choices(list(colors.keys()), weights=color_distribution[int(label)])
            color = colors[color_choice[0]]

            # Apply color to the grayscale image
            for i in range(3):  # There are three channels: R, G, B
                img_color[:, :, i] = (img_array * color[i] ).astype(np.uint8)

            # Convert to PIL Image and save
            img_pil = Image.fromarray(img_color)
            image_filename = f'image_{label}_{color_choice[0]}_{idx}.png'
            img_pil.save(os.path.join(base_dir, image_filename))

            # Write details to CSV
            csvwriter.writerow([image_filename, int(label), color_choice[0]])
 
# Load MNIST data
def main(dataset_type='train'):
    mnist_data = datasets.MNIST(root='.', train=(dataset_type=='train'), download=True, transform=transforms.ToTensor())
    
    # Define color distribution
    # color_distribution = {
    #     0: [0.9, 0.05, 0.05],
    #     1: [0.05, 0.9, 0.05],
    #     2: [0.05, 0.05, 0.9],
    #     3: [0.05, 0.9, 0.05],
    #     4: [0.9, 0.05, 0.05],
    #     5: [0.05, 0.9, 0.05],
    #     6: [0.05, 0.05, 0.9],
    #     7: [0.05, 0.9, 0.05],
    #     8: [0.9, 0.05, 0.05],
    #     9: [0.05, 0.05, 0.9],
    # }
    # train
    color_distribution = {
    0: [0.333, 0.333, 0.334],
    1: [0.333, 0.334, 0.333],
    2: [0.334, 0.333, 0.333],
    3: [0.333, 0.333, 0.334],
    4: [0.333, 0.334, 0.333],
    5: [0.334, 0.333, 0.333],
    6: [0.333, 0.333, 0.334],
    7: [0.333, 0.334, 0.333],
    8: [0.334, 0.333, 0.333],
    9: [0.333, 0.333, 0.334],
}
    # color_distribution = {
    #     0: [0.05, 0.9, 0.05],
    #     1: [0.05, 0.05, 0.9],
    #     2: [0.05, 0.9, 0.05],
    #     3: [0.9, 0.05, 0.05],
    #     4: [0.05, 0.9, 0.05],
    #     5: [0.05, 0.05, 0.9],
    #     6: [0.9, 0.05, 0.05],
    #     7: [0.05, 0.05, 0.9],
    #     8: [0.05, 0.9, 0.05],
    #     9: [0.9, 0.05, 0.05],
    # }

    # Colorize MNIST data
    colorize_mnist(list(zip(mnist_data.data, mnist_data.targets)), color_distribution, dataset_type)

if __name__ == '__main__':
    main('train')  # or 'test'
    print('Colorization complete.')

    # 读取CSV文件
    data = pd.read_csv('/local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv')

    # 计算每个label和颜色的组合数量
    counts = data.groupby(['label', 'color']).size().reset_index(name='count')

    # 打印结果
    print(counts)

 