#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=C
#SBATCH --output=C.txt  

source ~/.bashrc

conda activate llava-med
 
 

python CGAN.py \
   --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2 \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
   --genmodel "first_generation_gen.pt" \
   --num_epochs 50 \
   --n_critic 5 \
   --n_generator 5 \
   --display_step 100 \

echo "generation 1 "

python generate_images_test.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen1_images\
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv \
   --genmodel "first_generation_gen.pt" \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/generation1.csv \
   --gen_num 1 \
  
 


# echo "generation 1 20%"
 
# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
#    --genmodel "first_generation_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent/top20synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent/train.csv \
#    --toppercent 0.2


 