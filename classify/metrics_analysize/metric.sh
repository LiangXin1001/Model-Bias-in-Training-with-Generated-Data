#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=metric
#SBATCH --output=metric.txt  
 

source ~/.bashrc

conda activate llava-med



echo "gen0"
python tprfpr.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0
 
# python residuals.py\
#     --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0
 

python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/tpr_fpr_results.csv



echo "gen1"
python tprfpr.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1
 
# python residuals.py\
#     --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1
 

python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/tpr_fpr_results.csv



echo "gen2"
python tprfpr.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2
 
# python residuals.py\
#     --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/all_images_results.csv \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2
 

python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/tpr_fpr_results.csv


echo "gen3"
python tprfpr.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3
 
# python residuals.py\
#     --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/all_images_results.csv \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3

 

python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/tpr_fpr_results.csv




echo "gen4"
python tprfpr.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4
 
# python residuals.py\
#     --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/all_images_results.csv \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4




python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/tpr_fpr_results.csv




 