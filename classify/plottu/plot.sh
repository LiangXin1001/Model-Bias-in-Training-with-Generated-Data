#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot
#SBATCH --output=plot.txt  
 

source ~/.bashrc

conda activate llava-med


python ploteo.py

# python plot_digit_color_diff.py

python plotdi.py

# python plottpr.py
