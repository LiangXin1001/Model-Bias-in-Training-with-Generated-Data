#!/bin/bash
#SBATCH --job-name=training             
#SBATCH --gres=gpu:2                 
#SBATCH --mem=500G                   
#SBATCH --output=all.out       
#SBATCH --error=all.err        

source ~/.bashrc
 
conda activate llava-med
export CUDA_LAUNCH_BLOCKING=1
python mainACGAN.py