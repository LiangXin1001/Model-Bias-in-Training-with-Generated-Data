
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset
# 定义 CIFAR_100_CLASS_MAP 和 SuperCIFAR100 类
CIFAR_100_CLASS_MAP = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

class SuperCIFAR100(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.ds = torchvision.datasets.CIFAR100(**kwargs)
        self.classes = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        self.subclasses = self.ds.classes
        self.reverse_map = {}
        for i, k in enumerate(self.classes):
            for v_ in CIFAR_100_CLASS_MAP[k]:
                self.reverse_map[self.subclasses.index(v_)] = i
        # print("self.reverse_map : ",self.reverse_map)
        self.subclass_targets = self.ds.targets
        self.targets = [self.reverse_map[u] for u in self.ds.targets]
        # print("subclass_targets : ",self.subclass_targets)
        # print(" targets : ",self.targets)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        # print("x",x.shape," self.reverse_map[y] : ", self.reverse_map[y] , "  y : ",y)
        return x, self.reverse_map[y], y



class GeneratedDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (string): Directory with all the class folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_dataset()

    def load_dataset(self):
        """Loads dataset from file system."""
        for root_dir in self.root_dirs:
            # 遍历每一个分类的文件夹
            for label_id, class_dir in enumerate(sorted(os.listdir(root_dir))):
                class_folder = os.path.join(root_dir, class_dir)
                if os.path.isdir(class_folder):
                    # 遍历每个分类文件夹中的图像文件
                    for image_file in os.listdir(class_folder):
                        image_path = os.path.join(class_folder, image_file)
                        if os.path.isfile(image_path) and image_path.endswith('.png'):
                            self.images.append(image_path)
                            self.labels.append(label_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Fetches a single data point from the dataset."""
        image_path = self.images[idx]
        image = Image.open(image_path)   
        # image = read_image(image_path)
        label = self.labels[idx]
        image_path = self.images[idx]
   
        if self.transform:
            image = self.transform(image)
        return image, label, label  # 返回图像及其对应的两次大类标签
 

tf = transforms.Compose([transforms.Resize(64),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])


# import os
# import torch
# from torchvision.datasets import CIFAR100
# from torchvision.transforms import ToPILImage
# from tqdm import tqdm

# # 定义数据集目录和类别目录
# base_dir = 'CIFAR100_dataset'
# os.makedirs(base_dir, exist_ok=True)
# for super_class, sub_classes in CIFAR_100_CLASS_MAP.items():
#     super_class_dir = os.path.join(base_dir, super_class)
#     os.makedirs(super_class_dir, exist_ok=True)
#     for sub_class in sub_classes:
#         os.makedirs(os.path.join(super_class_dir, sub_class), exist_ok=True)

# # 加载 CIFAR-100 训练数据集
# dataset = CIFAR100(root='.', download=True, train=True)

# # 准备图像保存函数
# to_pil = ToPILImage()
 
# for idx in tqdm(range(len(dataset))):
#     image, label = dataset[idx]
#     class_name = dataset.classes[label]

#     for super_class, sub_classes in CIFAR_100_CLASS_MAP.items():
#         if class_name in sub_classes:
#             image_folder = os.path.join(base_dir, super_class, class_name)
#             image_path = os.path.join(image_folder, f'{idx}.png')
#             image.save(image_path)  # 直接保存，不需要转换
#             break


# print("图片保存完毕！")


