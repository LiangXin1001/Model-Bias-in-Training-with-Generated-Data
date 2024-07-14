#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=C
#SBATCH --output=C.txt  

source ~/.bashrc

conda activate llava-med
 
  

# python CGAN.py \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0 \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --genmodel "first_generation_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \

# echo "generation 1 "

# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
#    --genmodel "first_generation_gen.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
#    --gen_num 1 \
  

# # Source directory containing PNG images
# SOURCE_DIR="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_train"
# # Destination directory where images will be copied
# DEST_DIR="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train"

# # Create the destination directory if it doesn't already exist
# mkdir -p "$DEST_DIR"

# # Synchronize all PNG files from source to destination
# rsync -av --include '*/' --include '*.png' --exclude '*' "$SOURCE_DIR/" "$DEST_DIR/"



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




# echo "generation 1 50%"

# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
#    --genmodel "first_generation_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent/top50synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent/train.csv \
#    --toppercent 0.5


echo "generation 1 70%"
 
python make_next_gen_data.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
   --genmodel "first_generation_gen.pt" \
   --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
   --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent/top70synthetizedcsv.csv \
   --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent/train.csv \
   --toppercent 0.7


# echo "generation 1 80%"

# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
#    --genmodel "first_generation_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent/top80synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent/train.csv \
#    --toppercent 0.8

 
echo "generation 1 100%"
 
python make_next_gen_data.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
   --genmodel "first_generation_gen.pt" \
   --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/synthetizedtraincsv.csv \
   --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent/top100synthetizedcsv.csv \
   --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent/train.csv \
   --toppercent 1

# python CGAN.py \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --genmodel "second_generation_20percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \

# python CGAN.py \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#     --genmodel "second_generation_50percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \
python CGAN.py \
    --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent \
    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
    --genmodel "second_generation_70percent_gen.pt" \
    --num_epochs 50 \
    --n_critic 5 \
    --n_generator 5 \
    --display_step 100 \

# python CGAN.py \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --genmodel "second_generation_80percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \

python CGAN.py \
    --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent \
    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train \
    --genmodel "second_generation_100percent_gen.pt" \
    --num_epochs 50 \
    --n_critic 5 \
    --n_generator 5 \
    --display_step 100 \
