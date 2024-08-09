#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --job-name=plot
#SBATCH --output=plot.txt  
#SBATCH --mem=100G                   
source ~/.bashrc

conda activate llava-med
 
python gen_images.py
python fid.py