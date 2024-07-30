#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=clustering
#SBATCH --output=clustering.txt  
 

source ~/.bashrc

conda activate llava-med

python  hierarchical_clustering.py \
    --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0 \
 