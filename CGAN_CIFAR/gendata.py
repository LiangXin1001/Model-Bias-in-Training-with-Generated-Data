import torch
import torchvision.utils as vutils
import os
from model import Generator 
# 设定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
gen = Generator().to(device)
gen.load_state_dict(torch.load('gen.pth', map_location=device))

# 为每个类别生成1000张图像
num_classes = 20
num_images_per_class = 100
output_dir = 'generated_images'

os.makedirs(output_dir, exist_ok=True)

for label in range(num_classes):
    # 生成每个类别的图像
    noise = torch.randn(num_images_per_class, 100, device=device)
    labels = torch.full((num_images_per_class,), label, dtype=torch.long, device=device)
    with torch.no_grad():
        generated_images = gen(noise, labels)

    # 保存生成的图像
    for i, image in enumerate(generated_images):
        vutils.save_image(image, os.path.join(output_dir, f'class_{label}_image_{i:04d}.png'), normalize=True)

print("All images have been generated and saved.")
