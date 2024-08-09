#!/bin/bash
#SBATCH --job-name=training             
#SBATCH --gres=gpu:2                 
#SBATCH --mem=500G                   
#SBATCH --output=gendata.out       
#SBATCH --error=gendata.err        

source ~/.bashrc
 
conda activate llava-med
export CUDA_LAUNCH_BLOCKING=1

gennum=1
python mainACGAN.py \
    --gennum $gennum 

python gendata.py \
    --gennum $gennum  \
    --model_path  models 
 
python mainACGAN.py \
    --gennum 2 \
    --pkl_paths  data/generated_images_${gennum}.pkl 


gennum=2
