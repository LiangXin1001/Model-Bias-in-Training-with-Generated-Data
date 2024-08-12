import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import os
import argparse
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datasets import SuperCIFAR100 ,CIFAR_100_CLASS_MAP,tf,GeneratedDataset
from model import Generator, Discriminator
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
# torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
 
def parse_args():
    parser = argparse.ArgumentParser(description="Save models with custom filenames")
    parser.add_argument('--gennum', type=int, required=True, help='Generator number for filename customization')
    parser.add_argument('--data_root_paths', type=str, required=True, help='Generator number for filename customization')
    return parser.parse_args()

args = parse_args()
 

# prepare datasets
def load_and_prepare_datasets(pkl_paths, transform):
    images = []
    labels = []

    if pkl_paths:
        for pkl_path in pkl_paths.split(','):
            with open(pkl_path, 'rb') as f:
                imgs, lbls = pickle.load(f)
                images.extend(imgs)
                labels.extend(lbls)

        # 统计每个标签的图像数量
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1

        # 打印每个标签及其对应的图像数量
        index_to_superclass = {i: k for i, k in enumerate(sorted(CIFAR_100_CLASS_MAP.keys()))}
        for label, count in label_counts.items():
            print(f"Label {label} ({index_to_superclass[label]}): {count} images")

        generated_dataset = GeneratedDataset(images, labels)
    else:
        generated_dataset = None

    # 加载训练数据集
    trainset = SuperCIFAR100(root='./data', train=True, download=True, transform=tf)
    
    if generated_dataset:
        return torch.utils.data.ConcatDataset([trainset, generated_dataset])
    else:
        return trainset


def custom_collate_fn(batch):
    # 准备空列表来保存图像和标签
    images = []
    mapped_labels = []
    original_labels = []

    # 遍历批次中的每个数据项
    for item in batch:
        img, mapped_label, original_label = item
        
        # 添加到各自的列表
        images.append(img)
        mapped_labels.append(mapped_label)
        original_labels.append(original_label)

    # 使用 torch.stack 将图像列表转换为张量
    images = torch.stack(images)

    # 将标签列表转换为张量
    mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
    original_labels = torch.tensor(original_labels, dtype=torch.long)

    return images, mapped_labels, original_labels

 
 
  

# print("len(dataset)",len(dataset))
# print("dataset[0][0].size() ",dataset[0][0].size())
classes = sorted(CIFAR_100_CLASS_MAP.keys()) + ['fake']

 

def showImage(images,epoch=-99, idx = -99):
    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images,axes = (1,2,0)))
    plt.axis('off')
    if epoch!=-99:
        save_dir = "runs"
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建它
        save_path = os.path.join(save_dir, f"e{epoch}i{idx}.png")  # 创建文件路径
        plt.savefig(save_path)  # 保存文件到指定路径
        plt.close() 
        # plt.savefig("e" + str(epoch) + "i" + str(idx) + ".png")

# dataiter = iter(trainloader)
# images, mapped_labels, original_labels = next(dataiter)
 
# save_image(make_grid(images, nrow=8), 'batch_images.png')  # nrow 表示每行显示的图像数量
 
# print("images.size()",images.size())
# print("mapped_labels, original_labels")
# print(mapped_labels, original_labels)
 

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# dataset = load_and_prepare_datasets(args.pkl_paths, tf)

# trainloader = torch.utils.data.DataLoader(
#     dataset=dataset,  
#     batch_size=64,
#     shuffle=True,
#     num_workers=2 ,
#     collate_fn=custom_collate_fn   
# )
data_root_paths = args.data_root_paths.split(',') 
trainset = SuperCIFAR100(root='./data', train=True, download=True, transform=tf)
  
if args.gennum:
    generated_dataset = GeneratedDataset(root_dirs=data_root_paths, transform=tf)
    combined_dataset = ConcatDataset([generated_dataset, trainset])

    # 为训练创建 DataLoader
    trainloader = torch.utils.data.DataLoader(
        dataset=combined_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2 
    )
else:
    trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=64,
    shuffle=True,
    num_workers=2  
    )

gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

paramsG = list(gen.parameters())
print(len(paramsG))

paramsD = list(disc.parameters())
print(len(paramsD))        
        
optimG = optim.Adam(gen.parameters(), 0.0004, betas = (0.5,0.999))
optimD = optim.Adam(disc.parameters(), 0.0001, betas = (0.5,0.999))

validity_loss = nn.BCELoss()

# real_labels = 0.7 + 0.5 * torch.rand(10, device = device)
# fake_labels = 0.3 * torch.rand(10, device = device)
epochs = 40
for epoch in range(1,epochs+1):
    torch.cuda.empty_cache()
    for idx, (images,labels,_) in enumerate(trainloader,0):
        # print("idx",idx)
        # print(f"Batch {idx} has {images.size(0)} images.")
        batch_size = images.size(0)
        labels= labels.to(device)
        images = images.to(device)
        real_labels = 0.7 + 0.3 * torch.rand(batch_size, device=device)  # 现在产生的值介于 0.7 和 1.0 之间
        fake_labels = 0.3 * torch.rand(batch_size, device=device)        # 保持在 0 和 0.3 之间

       
        real_label = real_labels[idx % 10]
        fake_label = fake_labels[idx % 10]
        
        
        
        fake_class_labels = 20 *torch.ones((batch_size,),dtype = torch.long,device = device)
        if not (labels.max() < 21 and labels.min() >= 0):
            print("Invalid labels found:", labels[labels >= 21], labels[labels < 0])
        assert labels.max() < 21 and labels.min() >= 0, "Labels out of range"
        assert fake_class_labels.max() < 21 and fake_class_labels.min() >= 0, "Fake labels out of range"

        if idx % 25 == 0:
             real_label, fake_label = fake_label, real_label
        
        # ---------------------
        #         disc
        # ---------------------
        
        optimD.zero_grad()       
        
        # real
        validity_label = torch.full((batch_size,),real_label , device = device)
        # validity_label_real = torch.full((batch_size,), 1, device=device)  
        pvalidity, plabels = disc(images)       
        
        errD_real_val = validity_loss(pvalidity, validity_label)            
        errD_real_label = F.nll_loss(plabels,labels)
        
        errD_real = errD_real_val + errD_real_label
        errD_real.backward()
        
        D_x = pvalidity.mean().item()        
        
        #fake 
        noise = torch.randn(batch_size,100,device = device)  
        # print("noise : ",noise.shape)
        sample_labels = torch.randint(0,20,(batch_size,),device = device, dtype = torch.long)
        
        fakes = gen(noise,sample_labels)
        
        validity_label.fill_(fake_label)
        # validity_label_fake = torch.full((batch_size,), 0, device=device)  # 使用固定值0表示假冒

        pvalidity, plabels = disc(fakes.detach())       
        
        errD_fake_val = validity_loss(pvalidity, validity_label)
        errD_fake_label = F.nll_loss(plabels, fake_class_labels)
        
        errD_fake = errD_fake_val + errD_fake_label
        errD_fake.backward()
        
        D_G_z1 = pvalidity.mean().item()
        
        #finally update the params!
        errD = errD_real + errD_fake
        
        optimD.step()
    
        
        # ------------------------
        #      gen
        # ------------------------
        
        
        optimG.zero_grad()
        
        noise = torch.randn(batch_size,100,device = device)  
        sample_labels = torch.randint(0,20,(batch_size,),device = device, dtype = torch.long)
        
        validity_label.fill_(1)
        
        fakes = gen(noise,sample_labels)
        pvalidity,plabels = disc(fakes)
        
        errG_val = validity_loss(pvalidity, validity_label)        
        errG_label = F.nll_loss(plabels, sample_labels)
        
        errG = errG_val + errG_label
        errG.backward()
        
        D_G_z2 = pvalidity.mean().item()
        
        optimG.step()
        # # 在训练循环中添加检查点
        # print("Batch labels shape:", labels.shape)
        # print("Predicted labels shape:", plabels.shape)
        # print("Label values:", labels.unique())
        # print("Predicted validity shape:", pvalidity.shape,pvalidity)
        # print("Real validity label shape:", validity_label.shape,validity_label)
        # print("pvalidity range:", pvalidity.min().item(), pvalidity.max().item())
        # print("validity_label range:", validity_label.min().item(), validity_label.max().item())
        # print("pvalidity device:", pvalidity.device)
        # print("validity_label device:", validity_label.device)

        # # 检查数据类型和设备
        # print("Labels device and dtype:", labels.device, labels.dtype)
        # print("Plabels device and dtype:", plabels.device, plabels.dtype)

        
        print("[{}/{}] [{}/{}] D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] "
              .format(epoch,epochs, idx, len(trainloader),D_x, D_G_z1,D_G_z2,errG,errD,
                      errD_real_label + errD_fake_label + errG_label))
        
        
        if idx % 100 == 0:
             
            noise = torch.randn(20,100,device = device)  
            labels = torch.arange(0,20,dtype = torch.long,device = device)
            
            gen_images = gen(noise,labels).detach()
            
            showImage(make_grid(gen_images),epoch,idx)
       
     
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

 
gen_model_path = os.path.join(model_dir, f'gen_{args.gennum}.pth')
torch.save(gen.state_dict(), gen_model_path)
print(f"Generator model saved to {gen_model_path}")
 
disc_model_path = os.path.join(model_dir, f'disc_{args.gennum}.pth')
torch.save(disc.state_dict(), disc_model_path)
print(f"Discriminator model saved to {disc_model_path}")