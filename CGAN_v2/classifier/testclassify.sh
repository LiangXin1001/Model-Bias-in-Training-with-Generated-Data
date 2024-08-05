#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=testclass
#SBATCH --output=testclass.txt  
 

source ~/.bashrc

conda activate llava-med

python testclassify.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
   