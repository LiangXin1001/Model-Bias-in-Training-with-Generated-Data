#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=acc.txt  
#SBATCH --error=acc.err     
 
# 激活环境
source ~/.bashrc
conda activate llava-med

python metric/acc.py