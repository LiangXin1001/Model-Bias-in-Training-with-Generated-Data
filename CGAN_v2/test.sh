#!/bin/bash
#SBATCH --job-name=testing   # 作业名称
#SBATCH --ntasks-per-node=1           # 每节点一个任务
#SBATCH --cpus-per-task=4             # 每任务四个核心
#SBATCH --gres=gpu:1                  # 请求一个 GPU
#SBATCH --mem=16G                     # 内存大小
#SBATCH --output=test.out         # 标准输出文件
#SBATCH --error=test.err          # 标准错误文件

source ~/.bashrc
 
conda activate llava-med
  

echo "gen1"
python test.py \
    --num_images 2500 \
    --model-path weights_gen0/GAN-last.pth  \
    --output_csv  generated_images_gen1/gen1_test.csv  \
    --image_path generated_images_gen1\
    --gen_num 1


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen1\
    --csv_filename gen1_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  augmented_train_dataset\
    --generated_csv_path generated_images_gen1/gen1_test.csv \
    --new_csv_name combined_train_gen0_gen1.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen0_gen1.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen1"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."



# echo "gen2"
# python test.py \
#     --num_images 2500 \
#     --model-path weights_gen1/GAN-last.pth  \
#     --output_csv  generated_images_gen2/gen2_test.csv  \
#     --image_path generated_images_gen2\
#     --gen_num 2

# python   detect_color/detect_color.py\
#     --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen2\
#     --csv_filename gen2_test.csv
 
# python  augment/augment_dataset.py \
#     --train_csv_filename  combined_train_gen0_gen1.csv\
#     --generated_csv_path generated_images_gen2/gen2_test.csv \
#     --new_csv_name combined_train_gen1_gen2.csv
 
# python  augment/seedata.py\
#     --csv_filename combined_train_gen1_gen2.csv
 

 
# src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen2"
# dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2"
 
# mkdir -p "$dest_dir"
 
# find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

# echo "All images have been copied to $dest_dir."

# echo "gen3"
# python test.py \
#     --num_images 2500 \
#     --model-path weights_gen2/GAN-last.pth  \
#     --output_csv  generated_images_gen3/gen3_test.csv  \
#     --image_path generated_images_gen3\
#     --gen_num 3


# python   detect_color/detect_color.py\
#     --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen3\
#     --csv_filename gen3_test.csv
 
# python  augment/augment_dataset.py \
#     --train_csv_filename  combined_train_gen1_gen2.csv\
#     --generated_csv_path generated_images_gen3/gen3_test.csv \
#     --new_csv_name combined_train_gen2_gen3.csv
 
# python  augment/seedata.py\
#     --csv_filename combined_train_gen2_gen3.csv
 

 
# src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen3"
# dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3"
 
# mkdir -p "$dest_dir"
 
# find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

# echo "All images have been copied to $dest_dir."


# echo "gen4"
# python test.py \
#     --num_images 2500 \
#     --model-path weights_gen3/GAN-last.pth  \
#     --output_csv  generated_images_gen4/gen4_test.csv  \
#     --image_path generated_images_gen4\
#     --gen_num 4


# python   detect_color/detect_color.py\
#     --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen4\
#     --csv_filename gen4_test.csv
 
# python  augment/augment_dataset.py \
#     --train_csv_filename  combined_train_gen2_gen3.csv\
#     --generated_csv_path generated_images_gen4/gen4_test.csv \
#     --new_csv_name combined_train_gen3_gen4.csv
 
# python  augment/seedata.py\
#     --csv_filename combined_train_gen3_gen4.csv
 

 
# src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen4"
# dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4"
 
# mkdir -p "$dest_dir"
 
# find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

# echo "All images have been copied to $dest_dir."


# echo "gen5"
# python test.py \
#     --num_images 2500 \
#     --model-path weights_gen4/GAN-last.pth  \
#     --output_csv  generated_images_gen5/gen5_test.csv  \
#     --image_path generated_images_gen5\
#     --gen_num 5


# python   detect_color/detect_color.py\
#     --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen5\
#     --csv_filename gen5_test.csv
 
# python  augment/augment_dataset.py \
#     --train_csv_filename  combined_train_gen2_gen3.csv\
#     --generated_csv_path generated_images_gen5/gen5_test.csv \
#     --new_csv_name combined_train_gen4_gen5.csv
 
# python  augment/seedata.py\
#     --csv_filename combined_train_gen4_gen5.csv
 

 
# src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen5"
# dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5"
 
# mkdir -p "$dest_dir"
 
# find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

# echo "All images have been copied to $dest_dir."