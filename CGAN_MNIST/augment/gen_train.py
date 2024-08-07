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
    base_dir = f'MNIST/mnist_{dataset_type}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    csv_filename = os.path.join('MNIST', 'train.csv')
    # Prepare to write to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image_name', 'label', 'color'])
 
        # Dictionary to hold counters for each label
        label_counters = {i: 0 for i in range(10)}

        for image, label in data:
            # Convert image to numpy array
            img_array = np.array(image, dtype=np.float32) / 255
            # Prepare to colorize
            img_color = np.zeros((28, 28, 3), dtype=np.uint8)
            # Randomly decide the color based on the distribution for the given label
            color_choice = random.choices(list(colors.keys()), weights=color_distribution[int(label)])
            color = colors[color_choice[0]]
            # Apply color to the grayscale image
            for i in range(3):  # There are three channels: R, G, B
                img_color[:, :, i] = (img_array * color[i]).astype(np.uint8)
            
            # Get the current count for the label and increment it
            current_count = label_counters[int(label)]
            image_filename = f'image_{label}_{color_choice[0]}_{current_count}.png'
            label_counters[int(label)] += 1  # Increment the label counter

            # Convert to PIL Image and save
            img_pil = Image.fromarray(img_color)
            img_pil.save(os.path.join(base_dir, image_filename))

            # Write details to CSV
            csvwriter.writerow([image_filename, int(label), color_choice[0]])

# Load MNIST data
def main(dataset_type='train'):
    mnist_data = datasets.MNIST(root='.', train=(dataset_type=='train'), download=True, transform=transforms.ToTensor())
   
    # test
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
 

    # Colorize MNIST data
    colorize_mnist(list(zip(mnist_data.data, mnist_data.targets)), color_distribution, dataset_type)

if __name__ == '__main__':
    main('train')  # or 'test'
    print('Colorization complete.')
 
 