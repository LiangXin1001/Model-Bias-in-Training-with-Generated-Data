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
# python3 test.py \
#     --test_img_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --arch cgan \
#     --num-images 1000 \
#     --model-path weights_lr00003/GAN-last.pth  \
#     --output_csv  gen1_test_lr00003.csv  \
#     --image_path generated_images_lr00003


# python3 test.py \
#     --test_img_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --arch cgan \
#     --num-images 1000 \
#     --model-path weights_lr00005/GAN-last.pth  \
#     --output_csv  gen1_test_lr00005.csv   \
#     --image_path generated_images_lr00005

# python3 test.py \
#     --test_img_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --arch cgan \
#     --num-images 1000 \
#     --model-path weights_beta05/GAN-last.pth  \
#     --output_csv  gen1_test_beta05.csv  \
#     --image_path generated_images_beta05


python3 test.py \
    --test_img_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --arch cgan \
    --num-images 100 \
    --model-path weights_b16/GAN-last.pth  \
    --output_csv  gen1_test_b16.csv  \
    --image_path generated_images_b16

    