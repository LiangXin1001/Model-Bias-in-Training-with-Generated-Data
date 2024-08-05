#!/bin/bash
#SBATCH --job-name=training   # 作业名称
#SBATCH --ntasks-per-node=1           # 每节点一个任务
#SBATCH --cpus-per-task=4             # 每任务四个核心
#SBATCH --gres=gpu:1                  # 请求一个 GPU
#SBATCH --mem=16G                     # 内存大小
#SBATCH --output=all.out         # 标准输出文件
#SBATCH --error=all.err          # 标准错误文件

source ~/.bashrc
 
conda activate llava-med

python train.py  \
  --model_dir weights_gen0\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/augmented_train_dataset.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 


echo "gen1"
python test.py \
    --num_images 3000 \
    --model-path weights_gen0/GAN-last.pth  \
    --output_csv  generated_images_gen1/gen1_test.csv  \
    --image_path generated_images_gen1\
    --gen_num 1


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen1\
    --csv_filename gen1_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  augmented_train_dataset.csv\
    --generated_csv_path generated_images_gen1/gen1_test.csv \
    --new_csv_name combined_train_gen0_gen1.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen0_gen1.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen1"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."



python train.py  \
  --model_dir weights_gen1\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen0_gen1.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 




echo "gen2"
python test.py \
    --num_images 3700 \
    --model-path weights_gen1/GAN-last.pth  \
    --output_csv  generated_images_gen2/gen2_test.csv  \
    --image_path generated_images_gen2\
    --gen_num 2

python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen2\
    --csv_filename gen2_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen0_gen1.csv\
    --generated_csv_path generated_images_gen2/gen2_test.csv \
    --new_csv_name combined_train_gen1_gen2.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen1_gen2.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen2"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."

python train.py  \
  --model_dir weights_gen2\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen1_gen2.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2  \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 


echo "gen3"
python test.py \
    --num_images 5000 \
    --model-path weights_gen2/GAN-last.pth  \
    --output_csv  generated_images_gen3/gen3_test.csv  \
    --image_path generated_images_gen3\
    --gen_num 3


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen3\
    --csv_filename gen3_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen1_gen2.csv\
    --generated_csv_path generated_images_gen3/gen3_test.csv \
    --new_csv_name combined_train_gen2_gen3.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen2_gen3.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen3"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."



python train.py  \
  --model_dir weights_gen3\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen2_gen3.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 
 


echo "gen4"
python test.py \
    --num_images 7200 \
    --model-path weights_gen3/GAN-last.pth  \
    --output_csv  generated_images_gen4/gen4_test.csv  \
    --image_path generated_images_gen4\
    --gen_num 4


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen4\
    --csv_filename gen4_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen2_gen3.csv\
    --generated_csv_path generated_images_gen4/gen4_test.csv \
    --new_csv_name combined_train_gen3_gen4.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen3_gen4.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen4"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec cp {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."

 


python train.py  \
  --model_dir weights_gen4\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen3_gen4.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 
 


echo "gen5"
python test.py \
    --num_images 8000 \
    --model-path weights_gen4/GAN-last.pth  \
    --output_csv  generated_images_gen5/gen5_test.csv  \
    --image_path generated_images_gen5\
    --gen_num 5


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen5\
    --csv_filename gen5_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen3_gen4.csv\
    --generated_csv_path generated_images_gen5/gen5_test.csv \
    --new_csv_name combined_train_gen4_gen5.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen4_gen5.csv
 


 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen5"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."

echo "gen6"
python train.py  \
  --model_dir weights_gen5\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen4_gen5.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 
 
python test.py \
    --num_images 8000\
    --model-path weights_gen5/GAN-last.pth  \
    --output_csv  generated_images_gen6/gen6_test.csv  \
    --image_path generated_images_gen6\
    --gen_num 6


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen6\
    --csv_filename gen6_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen4_gen5.csv\
    --generated_csv_path generated_images_gen6/gen6_test.csv \
    --new_csv_name combined_train_gen5_gen6.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen5_gen6.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen6"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."





echo "gen7"

python train.py  \
  --model_dir weights_gen6\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen5_gen6.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights_gen6/GAN-last.pth  \
    --output_csv  generated_images_gen7/gen7_test.csv  \
    --image_path generated_images_gen7\
    --gen_num 7


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen7\
    --csv_filename gen7_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen5_gen6.csv\
    --generated_csv_path generated_images_gen7/gen7_test.csv \
    --new_csv_name combined_train_gen6_gen7.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen6_gen7.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen7"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."





echo "gen8"


python train.py  \
  --model_dir weights_gen7\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen6_gen7.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 
python test.py \
    --num_images 8000 \
    --model-path weights_gen7/GAN-last.pth  \
    --output_csv  generated_images_gen8/gen8_test.csv  \
    --image_path generated_images_gen8\
    --gen_num 8


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen8\
    --csv_filename gen8_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen6_gen7.csv\
    --generated_csv_path generated_images_gen8/gen8_test.csv \
    --new_csv_name combined_train_gen7_gen8.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen7_gen8.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen8"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."






echo "gen9"



python train.py  \
  --model_dir weights_gen8\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen7_gen8.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 
python test.py \
    --num_images 8000 \
    --model-path weights_gen8/GAN-last.pth  \
    --output_csv  generated_images_gen9/gen9_test.csv  \
    --image_path generated_images_gen9\
    --gen_num 9


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen9\
    --csv_filename gen9_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen7_gen8.csv\
    --generated_csv_path generated_images_gen9/gen9_test.csv \
    --new_csv_name combined_train_gen8_gen9.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen8_gen9.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen9"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen9"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."




echo "gen10"


python train.py  \
  --model_dir weights_gen9\
  --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen8_gen9.csv  \
  --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen9 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights_gen9/GAN-last.pth  \
    --output_csv  generated_images_gen10/gen10_test.csv  \
    --image_path generated_images_gen10\
    --gen_num 10


python   detect_color/detect_color.py\
    --base_image_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen10\
    --csv_filename gen10_test.csv
 
python  augment/augment_dataset.py \
    --train_csv_filename  combined_train_gen8_gen9.csv\
    --generated_csv_path generated_images_gen10/gen10_test.csv \
    --new_csv_name combined_train_gen9_gen10.csv
 
python  augment/seedata.py\
    --csv_filename combined_train_gen9_gen10.csv
 

 
src_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/generated_images_gen10"
dest_dir="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen10"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."