import torch
import clip
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):  # 遍历文件夹
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 确保处理的是图像文件
            img_path = os.path.join(folder, filename)  # 构建完整的文件路径
            try:
                with Image.open(img_path) as img:  # 使用with语句确保文件正确关闭
                    images.append(img.copy())  # 加载图像并复制，确保img对象可以正确关闭
            except IOError:
                print(f"Cannot open image {filename}")
    return images
# 图像预处理和特征提取
def get_clip_features(images):
    images_preprocessed = torch.stack([preprocess(image).unsqueeze(0).to(device) for image in images])
    with torch.no_grad():
        features = model.encode_image(images_preprocessed)
    return features.cpu().numpy()
real_image_paths = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train'
fake_image_paths = '/local/scratch/hcui25/Project/xin/CS/GAN/testQuality/gen0'
real_images = load_images_from_folder(real_image_paths)
fake_images = load_images_from_folder(fake_image_paths)
  
real_features = get_clip_features(real_images)
fake_features = get_clip_features(fake_images)

# 使用t-SNE降维到二维
tsne = TSNE(n_components=2, random_state=42)
all_features = np.vstack([real_features, fake_features])
all_features_2d = tsne.fit_transform(all_features)

# 可视化
plt.figure(figsize=(8, 8))
plt.scatter(all_features_2d[:len(real_features), 0], all_features_2d[:len(real_features), 1], color='blue', label='Real')
plt.scatter(all_features_2d[len(real_features):, 0], all_features_2d[len(real_features):, 1], color='red', label='Fake')
plt.legend()
plt.tight_layout()

output_path = 'gen0.png'
plt.savefig(output_path)
plt.close()

 
