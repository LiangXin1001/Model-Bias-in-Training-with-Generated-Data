# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
# %%



import torch
import torchvision

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


def getdDataset(opt):

    if opt.dataset == 'mnist':
        dst = datasets.MNIST(
            # 相对路径，以调用的文件位置为准
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'fashion':
        dst = datasets.FashionMNIST(
            root=opt.dataroot,
            train=True,
            download=True,
            # split='mnist',
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'cifar10':
        dst = datasets.CIFAR10(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'cifar100':
        dst = SuperCIFAR100(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )

    dataloader = DataLoader(
        dst,
        batch_size=opt.batch_size, 
        shuffle=True,
    )

    return dataloader

# %%
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    class opt:
        dataroot = '../../data'
        dataset = 'mnist'
        img_size = 32
        batch_size = 10

    dataloader = getdDataset(opt)
    for i, (imgs, labels) in enumerate(dataloader):
        print(i, imgs.shape, labels.shape)
        print(labels)

        img = imgs[0]
        img = img.numpy()
        img = make_grid(imgs, normalize=True).numpy()
        img = np.transpose(img, (1, 2, 0))

        plt.imshow(img)
        plt.show()
        plt.close()
        break
# %%
