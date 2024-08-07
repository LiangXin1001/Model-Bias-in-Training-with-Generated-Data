#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot
#SBATCH --output=plot.txt  
 

source ~/.bashrc

conda activate llava-med
 
SCRIPTS_DIR="./plottu/"


python ${SCRIPTS_DIR}average_acc.py
python ${SCRIPTS_DIR}ploteo.py

python ${SCRIPTS_DIR}plot_digit_color_diff.py

python ${SCRIPTS_DIR}plotdi.py

python ${SCRIPTS_DIR}plottpr.py

python ${SCRIPTS_DIR}digit_difference.py