#!/bin/bash
#SBATCH --job-name=testing   # 作业名称
#SBATCH --ntasks-per-node=1           # 每节点一个任务
#SBATCH --cpus-per-task=4             # 每任务四个核心
#SBATCH --gres=gpu:1                  # 请求一个 GPU
#SBATCH --mem=16G                     # 内存大小
#SBATCH --output=test.out         # 标准输出文件
#SBATCH --error=test.err          # 标准错误文件

source ~/.bashrc
 
conda activate llava-med
  

echo "gen1"
python test.py \
    --num_images 2500 \
    --model-path weights_gen0/GAN-last.pth  \
    --output_csv  generated_images_gen1/gen1_test.csv  \
    --image_path generated_images_gen1\
    --gen_num 1

 