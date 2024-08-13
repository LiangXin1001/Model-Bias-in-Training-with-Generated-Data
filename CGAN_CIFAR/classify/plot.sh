#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plot
#SBATCH --output=plot.txt  
 

source ~/.bashrc

conda activate llava-med
model_name="resnet50"
python plottu/average_acc.py  --model_name $model_name
python plottu/plottpr.py  --model_name $model_name
python plottu/ploteo.py --model_name $model_name

python plottu/plotdi.py --model_name $model_name
python plottu/plot_digit_color_diff.py --model_name $model_name

  
model_name="vgg19"
python plottu/average_acc.py  --model_name $model_name
python plottu/plottpr.py  --model_name $model_name
python plottu/ploteo.py --model_name $model_name

python plottu/plotdi.py --model_name $model_name
python plottu/plot_digit_color_diff.py --model_name $model_name

model_name="alexnet"
python plottu/average_acc.py  --model_name $model_name
python plottu/plottpr.py  --model_name $model_name
python plottu/ploteo.py --model_name $model_name

python plottu/plotdi.py --model_name $model_name
python plottu/plot_digit_color_diff.py --model_name $model_name


echo "all figures"

python plottu/acc_for_multiplemodals.py --model_names "vgg19" "resnet50" "alexnet"
