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
 
python residuals.py\
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0


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
 
python residuals.py\
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1


python disparate_impact.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1
 

python EO.py \
    --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv \
    --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/tpr_fpr_results.csv

for i in {2..10}
do
  echo "gen${i}"
  
  python tprfpr.py \
      --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}/all_images_results.csv \
      --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}
 
  python residuals.py \
      --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}/all_images_results.csv \
      --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}

  python disparate_impact.py \
      --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}/all_images_results.csv \
      --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}

  python EO.py \
      --result_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}/all_images_results.csv \
      --tprfpr_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen${i}/tpr_fpr_results.csv
done 
 