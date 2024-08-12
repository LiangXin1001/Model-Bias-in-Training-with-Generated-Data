#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=acc.txt  
#SBATCH --error=acc.err     
 
# 激活环境
source ~/.bashrc
conda activate llava-med


model_name="resnet50"
python metric/acc.py

python metric/tprfpr.py --result_dir results/${model_name} --model_name ${model_name}
 
# python metric/calculate_eo.py --result_dir  results/${model_name} --model_name ${model_name}


# python metric/calculate_di.py --result_dir  results/${model_name} --model_name ${model_name}
