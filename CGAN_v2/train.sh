#!/bin/bash
#SBATCH --job-name=training   # 作业名称
#SBATCH --ntasks-per-node=1           # 每节点一个任务
#SBATCH --cpus-per-task=4             # 每任务四个核心
#SBATCH --gres=gpu:1                  # 请求一个 GPU
#SBATCH --mem=16G                     # 内存大小
#SBATCH --output=train.out         # 标准输出文件
#SBATCH --error=train.err          # 标准错误文件

source ~/.bashrc
 
conda activate llava-med

python train.py  \
  --model_dir weights_gen0\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/augmented_train_dataset.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

# python train.py  \
#   --model_dir weights_gen1\
#   --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen0_gen1.csv  \
#   --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1 \
#   --arch cgan \
#   --epochs 50 \
#   --start-epoch 0 \
#   --b 16 \
#   --lr 0.0004 \
#   --image-size 28 \
#   --channels 3 

# python train.py  \
#   --model_dir weights_gen2\
#   --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen1_gen2.csv  \
#   --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2  \
#   --arch cgan \
#   --epochs 50 \
#   --start-epoch 0 \
#   --b 16 \
#   --lr 0.0004 \
#   --image-size 28 \
#   --channels 3 

# python train.py  \
#   --model_dir weights_gen3\
#   --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen2_gen3.csv  \
#   --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3 \
#   --arch cgan \
#   --epochs 50 \
#   --start-epoch 0 \
#   --b 16 \
#   --lr 0.0004 \
#   --image-size 28 \
#   --channels 3 
 
# python train.py  \
#   --model_dir weights_gen4\
#   --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen3_gen4.csv  \
#   --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4 \
#   --arch cgan \
#   --epochs 50 \
#   --start-epoch 0 \
#   --b 16 \
#   --lr 0.0004 \
#   --image-size 28 \
#   --channels 3 
 