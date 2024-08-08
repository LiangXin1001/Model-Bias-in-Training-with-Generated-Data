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
  --model_dir model/weights_gen0\
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 64 \
  --lr 0.0004 \
  --image-size 32 \
  --channels 3 
 