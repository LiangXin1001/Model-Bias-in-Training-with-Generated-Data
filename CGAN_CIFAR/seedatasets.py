import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
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
def save_image(img, filename):
    img = img / 2 + 0.5  # 逆归一化
    npimg = img.numpy()
    plt.imsave(filename, np.transpose(npimg, (1, 2, 0)))

class SuperCIFAR100(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.ds = torchvision.datasets.CIFAR100(**kwargs)
        self.classes = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        self.subclasses = self.ds.classes
        self.reverse_map = {}
        for i, k in enumerate(self.classes):
            for v_ in CIFAR_100_CLASS_MAP[k]:
                self.reverse_map[self.subclasses.index(v_)] = i
        self.subclass_targets = self.ds.targets
        self.targets = [self.reverse_map[u] for u in self.ds.targets]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, self.reverse_map[y], y

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset = SuperCIFAR100(root='./data', train=True, download=True, transform=transform)


def custom_collate_fn(batch):
    # 分离图像、映射标签、原始标签
    images, mapped_labels, original_labels = zip(*batch)
    
    # 将图像堆叠成一个新的张量
    images = torch.stack(images)
    
    # 将标签转换为张量
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
    original_labels = torch.tensor(original_labels, dtype=torch.long)
    
    # 返回处理后的批次数据
    return images, mapped_labels, original_labels

# 在创建 DataLoader 时指定自定义的 collate_fn
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,  
    batch_size=64,
    shuffle=True,
    num_workers=2,
    collate_fn=custom_collate_fn   
)




# 创建数据加载器
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# 获取一批数据
dataiter = iter(data_loader)
images, super_labels, sub_labels =  next(dataiter)

# 显示图像及其标签
def imshow(img):
    img = img / 2 + 0.5  # 逆归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 打印 Super Classes
print(f'Total number of Super Classes: {len(CIFAR_100_CLASS_MAP)}')
print('Super Classes:')
for super_class in CIFAR_100_CLASS_MAP.keys():
    print(super_class)

# 打印 Sub Classes
sub_classes = [sub_class for sub_classes in CIFAR_100_CLASS_MAP.values() for sub_class in sub_classes]
print(f'\nTotal number of Sub Classes: {len(sub_classes)}')
print('Sub Classes:')
for sub_class in sub_classes:
    print(sub_class)
# 保存图像
save_image(torchvision.utils.make_grid(images), 'cifar100_sample.png')