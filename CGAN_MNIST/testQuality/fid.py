import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image
import os

def load_images_from_folder(folder, max_images=None):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')) and (max_images is None or len(images) < max_images):  # 确保只处理图像文件
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)
        if max_images and len(images) >= max_images:
            break
    return images

def get_features(images, model, transform, device, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        if not batch:
            continue  # 如果批次为空，跳过这个批次
        image_tensors = torch.stack([transform(image).to(device) for image in batch])
        with torch.no_grad():
            output = model(image_tensors)
            features.append(output.detach().cpu().numpy())
    if features:
        return np.concatenate(features, axis=0)
    else:
        raise ValueError("No features were extracted. Check the input images and model processing steps.")


def calculate_fid(real_images, fake_images, device='cuda', batch_size=32):
    # 加载Inception模型
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 获取真实图像和生成图像的特征
    real_features = get_features(real_images, model, transform, device, batch_size)
    fake_features = get_features(fake_images, model, transform, device, batch_size)

    # 计算每个特征集的均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    # 计算FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid
 
real_image_paths = ' ../MNIST/mnist_test'
base_fake_image_path = '../gen'
  
for i in range(10):  # 从 gen0 到 gen9
    fake_image_paths = f"{base_fake_image_path}{i}"
    real_images = load_images_from_folder(real_image_paths, max_images=10000)
    fake_images = load_images_from_folder(fake_image_paths, max_images=10000)

    fid_score = calculate_fid(real_images, fake_images, device='cuda', batch_size=32)
    print(f"FID score for {fake_image_paths}: {fid_score}")