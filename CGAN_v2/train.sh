#!/bin/bash
#SBATCH --job-name=training   # 作业名称
#SBATCH --ntasks-per-node=1           # 每节点一个任务
#SBATCH --cpus-per-task=4             # 每任务四个核心
#SBATCH --gres=gpu:1                  # 请求一个 GPU
#SBATCH --mem=16G                     # 内存大小
#SBATCH --output=train16.out         # 标准输出文件
#SBATCH --error=train16.err          # 标准错误文件

source ~/.bashrc
 
conda activate llava-med

python train.py \
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
  --arch cgan \
  --epochs 40 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 \
 