#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=C
#SBATCH --output=C1e-5.txt  

source ~/.bashrc

conda activate llava-med
# python CGAN.py \
#    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
#    --genmodel "first_generation_gen_50epoch-5.pt" \
#    --num_epochs 50 \
#    --n_critic 5 \
#    --n_generator 5 \
#    --display_step 100 \

# echo "generation 1 "

# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen1_images50-5\
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
#    --genmodel "first_generation_gen_50epoch-5.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/generation50-5.csv \
 
 
# python sample_and_add_color.py \
#    --input_csv1 /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/generation50-5.csv \
#    --input_csv2 /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv \
#    --output_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen1_50-5_train.csv \
#    --num_samples 500


# python CGAN.py \
#    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen1_50-5_train.csv\
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen1_images50-5  \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
#    --genmodel "second_generation_gen_50-5epoch.pt" \
#    --num_epochs 50 \
#    --n_critic 5 \
#    --n_generator 5 \
#    --display_step 100 \


 
# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen2_images50-5\
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
#    --genmodel "second_generation_gen_50-5epoch.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/second_generation50-5.csv \
 


python sample_and_add_color.py \
   --input_csv1 /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/second_generation50-5.csv \
   --input_csv2 /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/train.csv \
   --output_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen2_50-5_train.csv \
   --num_samples 500


python CGAN.py \
   --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen2_50-5_train.csv \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen2_images50-5 \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
   --genmodel "third_generation_gen_50-5poch.pt" \
   --num_epochs 50 \
   --n_critic 5 \
   --n_generator 5 \
   --display_step 100 \

echo "generation 3 "

python generate_images_test.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/gen3_images50-5\
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/model \
   --genmodel "third_generation_gen_50-5epoch.pt" \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/third_generation50-5.csv \
    