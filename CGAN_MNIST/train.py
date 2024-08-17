# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging
import os
import random
import time
import warnings
 
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision.transforms import ToTensor
 
import cgan_pytorch.models as models
from cgan_pytorch.models.discriminator import discriminator_for_mnist
from cgan_pytorch.utils.common import AverageMeter
from cgan_pytorch.utils.common import ProgressMeter
from cgan_pytorch.utils.common import configure
from cgan_pytorch.utils.common import create_folder
# import wandb
from utils import utils
# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径,包含图片名称和标签。
            img_dirs (string): 包含所有图片的目录路径。
            transform (callable, optional): 可选的转换函数，应用于样本。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        image = None
        # 在所有目录中查找图像
        for img_dir in self.img_dirs:
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                break
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found in any of the directories.")
        
        label = self.img_labels.iloc[idx, 1]
        color = self.img_labels.iloc[idx, 2]  # 加载颜色标签
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)  # 默认转换为张量
        return image, label  # 返回图像、数值标签和颜色标签



 
def main(args):
    print(args)
  
    # wandb.init(project="DCGANnew_h100", config=args.__dict__)

    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True
 
    logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")
 
    ngpus_per_node = torch.cuda.device_count()
    
    main_worker(ngpus_per_node, args)

 
def main_worker(ngpus_per_node, args):
    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    # create model
    generator = configure(args)
    discriminator = discriminator_for_mnist(args.image_size, args.channels)
    print("create model done")
    discriminator = discriminator.to(device)
    generator = generator.to(device)
    # Loss of original GAN paper.
    adversarial_criterion = nn.MSELoss().cuda()

    fixed_noise = torch.randn([args.batch_size, 100])
    fixed_conditional = torch.randint(0, 10, (args.batch_size,))
     
    fixed_noise = fixed_noise.cuda()
    fixed_conditional = fixed_conditional.cuda()

    # All optimizer function and scheduler function.
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Selection of appropriate treatment equipment.
    train_csv =  args.train_csv
    image_dirs = args.image_dirs.split(',')
 
    mean_rgb , std_rgb = utils.get_mean_std(train_csv, image_dirs)
 
    train_dataset = CustomMNISTDataset(
        csv_file=args.train_csv, 
        img_dirs=args.image_dirs.split(','),  # 如果有多个目录，确保正确分割
        transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 调整为合适的归一化参数
        ])
    )

  
    sampler = None
   
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(sampler is None),
                                             pin_memory=True,
                                             sampler=sampler,
                                             num_workers=args.workers)

    # Load pre training model.
    if args.netD != "":
        discriminator.load_state_dict(torch.load(args.netD))
    if args.netG != "":
        generator.load_state_dict(torch.load(args.netG))
 
    for epoch in range(args.start_epoch, args.epochs):
    
        batch_time = AverageMeter("Time", ":6.4f")
        d_losses = AverageMeter("D Loss", ":6.6f")
        g_losses = AverageMeter("G Loss", ":6.6f")
        d_x_losses = AverageMeter("D(x)", ":6.6f")
        d_g_z1_losses = AverageMeter("D(G(z1))", ":6.6f")
        d_g_z2_losses = AverageMeter("D(G(z2))", ":6.6f")

        progress = ProgressMeter(num_batches=len(dataloader),
                                 meters=[batch_time, d_losses, g_losses, d_x_losses, d_g_z1_losses, d_g_z2_losses],
                                 prefix=f"Epoch: [{epoch}]")

        # Switch to train mode.

        discriminator.train()
        generator.train()
       
        end = time.time()
        for i, (inputs, target) in enumerate(dataloader):
            # Move data to special device.
            # print(inputs.shape, target.shape)
            inputs = inputs.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
            batch_size = inputs.size(0)
            # print("inputs.size(0) :  ", inputs.size(0) )
                    
            # batch_size = args.batch_size
            # The real sample label is 1, and the generated sample label is 0.
            real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype).cuda(device, non_blocking=True)
            fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype).cuda(device, non_blocking=True)
           
            noise = torch.randn([batch_size, 100])
            conditional = torch.randint(0, 10, (batch_size,))
            # Move data to special device.
 
            noise = noise.cuda(device, non_blocking=True)
            # print("noise,", noise)
            conditional = conditional.cuda(device, non_blocking=True)

            ##############################################
            # (1) Update D network: max E(x)[log(D(x))] + E(z)[log(1- D(z))]
            ##############################################
            # Set discriminator gradients to zero.
            discriminator.zero_grad()

            # Train with real.
            real_output = discriminator(inputs, target)
            # print("Real output size: ", real_output.size())
            # print("Real label size: ", real_label.size())
            d_loss_real = adversarial_criterion(real_output, real_label)
            
            d_loss_real.backward()
            d_x = real_output.mean()

            # Train with fake.
            fake = generator(noise, conditional)
            fake_output = discriminator(fake.detach(), conditional)
            d_loss_fake = adversarial_criterion(fake_output, fake_label)
            d_loss_fake.backward()
            d_g_z1 = fake_output.mean()

            # Count all discriminator losses.
            d_loss = d_loss_real + d_loss_fake
            discriminator_optimizer.step()

            ##############################################
            # (2) Update G network: min E(z)[log(1- D(z))]
            ##############################################
            # Set generator gradients to zero.
           
            generator.zero_grad()

            fake_output = discriminator(fake, conditional)
            g_loss = adversarial_criterion(fake_output, real_label)
           
            g_loss.backward()   
         
            d_g_z2 = fake_output.mean()
            generator_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            d_losses.update(d_loss.item(), inputs.size(0))
            g_losses.update(g_loss.item(), inputs.size(0))
            d_x_losses.update(d_x.item(), inputs.size(0))
            d_g_z1_losses.update(d_g_z1.item(), inputs.size(0))
            d_g_z2_losses.update(d_g_z2.item(), inputs.size(0))

            iters = i + epoch * len(dataloader) + 1
   
            # writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            # writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            # writer.add_scalar("Train/D_x", d_x.item(), iters)
            # writer.add_scalar("Train/D_G_z1", d_g_z1.item(), iters)
            # writer.add_scalar("Train/D_G_z2", d_g_z2.item(), iters)
            # Logging to wandb
            # wandb.log({
            #     "Train/D_Loss": d_loss.item(),
            #     "Train/G_Loss": g_loss.item(),
            #     "Train/D_x": d_x.item(),
            #     "Train/D_G_z1": d_g_z1.item(),
            #     "Train/D_G_z2": d_g_z2.item(),
            #     "Epoch": epoch,
            #     "Batch": i
            # })
            # Output results every 100 batches.
            if i % 100 == 0:
                progress.display(i)

        # Each Epoch validates the model once.
        with torch.no_grad():
            # Switch model to eval mode.
            generator.eval()
            sr = generator(fixed_noise, fixed_conditional)
            vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"), normalize=True)
        
        os.makedirs(args.model_dir, exist_ok=True)
        print("generator : ", generator)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save(generator.state_dict(), os.path.join(args.model_dir, f"Generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.model_dir, f"Discriminator_epoch{epoch}.pth"))

    torch.save(generator.state_dict(), os.path.join(args.model_dir, f"GAN-last.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_dir = "weights_gen4"

    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--image_dirs', type=str, required=True)
    parser.add_argument("--arch", default="cgan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `cgan`)")
    
    parser.add_argument("--workers", default=4, type=int,
                        help="Number of data loading workers. (Default: 4)")
  
    parser.add_argument("--epochs", default=128, type=int,
                        help="Number of total epochs to run. (Default: 128)")
    parser.add_argument("--start-epoch", default=0, type=int,
                        help="Manual epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("-b", "--batch_size", default=64, type=int,
                        help="The batch size of the dataset. (Default: 64)")
    parser.add_argument("--lr", default=0.0002, type=float,
                        help="Learning rate. (Default: 0.0002)")
    parser.add_argument("--image-size", default=28, type=int,
                        help="Image size of high resolution image. (Default: 28)")
    parser.add_argument("--channels", default=1, type=int,
                        help="The number of channels of the image. (Default: 1)")
    parser.add_argument("--netD", default="", type=str,
                        help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG", default="", type=str,
                        help="Path to Generator checkpoint.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="Number of nodes for distributed training.")
    parser.add_argument("--rank", default=-1, type=int,
                        help="Node rank for distributed training. (Default: -1)")
    parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                        help="url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)")
    parser.add_argument("--dist-backend", default="nccl", type=str,
                        help="Distributed backend. (Default: `nccl`)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--multiprocessing-distributed", action="store_true",
                        help="Use multi-processing distributed training to launch "
                             "N processes per node, which has N GPUs. This is the "
                             "fastest way to use PyTorch for either single node or "
                             "multi node data parallel training.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Training Engine.\n")

    create_folder("runs")
    create_folder(args.model_dir)

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.06.02")
    print("##################################################\n")

    main(args)

    logger.info("All training has been completed successfully.\n")
