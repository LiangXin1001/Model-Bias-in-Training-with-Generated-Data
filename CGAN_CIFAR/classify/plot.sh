#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot
#SBATCH --output=plot.txt  
 

source ~/.bashrc

conda activate llava-med
model_name="resnet50"
python plottu/average_acc.py  --model_name $model_name
python plottu/plottpr.py  --model_name $model_name
# python ploteo.py

# python plot_digit_color_diff.py

# python plotdi.py



# python digit_difference.py