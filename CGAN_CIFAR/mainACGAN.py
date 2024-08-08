import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from cgan_pytorch.utils.datasets import SuperCIFAR100  ,CIFAR_100_CLASS_MAP
from torchvision.utils import save_image
# torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


tf = transforms.Compose([transforms.Resize(64),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

 
 
# 修改这部分代码以使用你的 SuperCIFAR100 类
trainset = SuperCIFAR100(root='./data', train=True, download=True, transform=tf)
testset = SuperCIFAR100(root='./data', train=False, download=True, transform=tf)
 
dataset = torch.utils.data.ConcatDataset([trainset, testset])

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

 
trainloader = torch.utils.data.DataLoader(
    dataset=dataset,  
    batch_size=64,
    shuffle=True,
    num_workers=0 ,
    collate_fn=custom_collate_fn   
)

 
  

print("len(dataset)",len(dataset))
print("dataset[0][0].size() ",dataset[0][0].size())
classes = sorted(CIFAR_100_CLASS_MAP.keys()) + ['fake']

 

def showImage(images,epoch=-99, idx = -99):
    images = images.cpu().numpy()
    images = images/2 + 0.5
    plt.imshow(np.transpose(images,axes = (1,2,0)))
    plt.axis('off')
    if epoch!=-99:
        plt.savefig("e" + str(epoch) + "i" + str(idx) + ".png")

# dataiter = iter(trainloader)
# images, mapped_labels, original_labels = next(dataiter)
 
# save_image(make_grid(images, nrow=8), 'batch_images.png')  # nrow 表示每行显示的图像数量
 
# print("images.size()",images.size())
# print("mapped_labels, original_labels")
# print(mapped_labels, original_labels)
 
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator,self).__init__()
        
        #input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100,512,4,1,0,bias = False),
                                   nn.ReLU(True))

        #input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        #input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        #input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        #input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1,bias = False),
                                   nn.Tanh())
        #output 3*64*64
      
        self.embedding = nn.Embedding(20,100)
        
        
    def forward(self,noise,label):
        
        label_embedding = self.embedding(label)
        x = torch.mul(noise,label_embedding)
        x = x.view(-1,100,1,1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
        

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()        
        
        #input 3*64*64
        self.layer1 = nn.Sequential(nn.Conv2d(3,64,4,2,1,bias = False),
                                    nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        
        #input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64,128,4,2,1,bias = False),
                                    nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        #input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128,256,4,2,1,bias = False),
                                    nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout2d(0.5))
        #input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256,512,4,2,1,bias = False),
                                    nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.2,True))
        #input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512,1,4,1,0,bias = False),
                                   nn.Sigmoid())
        
        self.label_layer = nn.Sequential(nn.Conv2d(512,21,4,1,0,bias = False),
                                   nn.LogSoftmax(dim = 1))
        
    def forward(self,x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)
        
        validity = validity.view(-1)
        plabel = plabel.view(-1,21)
        
        return validity,plabel


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

paramsG = list(gen.parameters())
print(len(paramsG))

paramsD = list(disc.parameters())
print(len(paramsD))        
        
optimG = optim.Adam(gen.parameters(), 0.0002, betas = (0.5,0.999))
optimD = optim.Adam(disc.parameters(), 0.0002, betas = (0.5,0.999))

validity_loss = nn.BCELoss()

# real_labels = 0.7 + 0.5 * torch.rand(10, device = device)
# fake_labels = 0.3 * torch.rand(10, device = device)
epochs = 10

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
       
    
torch.save(gen.state_dict(),'gen.pth')
torch.save(disc.state_dict(),'disc.pth')