#!/bin/bash
#SBATCH --job-name=training   
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=4             
#SBATCH --gres=gpu:1                  
#SBATCH --mem=16G                    
#SBATCH --output=all.out         
#SBATCH --error=all.err        

source ~/.bashrc
 
conda activate llava-med

python augment/gen_test.py
python augment/gen_train.py
python augment/normalize_label_counts.py
python augment/augment_images_with_rotation.py
  
  
python train.py  \
  --model_dir weights/weights_gen0\
  --train_csv augment/augmented_train_dataset.csv  \
  --image_dirs MNIST/mnist_train \
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
    --model-path weights/weights_gen0/GAN-last.pth  \
    --output_csv data/gen1_test.csv  \
    --image_path generated_images_gen1\
    --gen_num 1

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen1\
    --csv_filename data/gen1_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename augment/augmented_train_dataset.csv\
    --generated_csv_path data/gen1_test.csv \
    --new_csv_name data/combined_train_gen0_gen1.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen0_gen1.csv

src_dir="generated_images_gen1"
dest_dir="data/generated_images_gen1"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

python train.py  \
  --model_dir weights/weights_gen1\
  --train_csv data/combined_train_gen0_gen1.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1 \
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
    --model-path weights/weights_gen1/GAN-last.pth  \
    --output_csv data/gen2_test.csv  \
    --image_path generated_images_gen2\
    --gen_num 2

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen2\
    --csv_filename data/gen2_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen0_gen1.csv\
    --generated_csv_path data/gen2_test.csv \
    --new_csv_name data/combined_train_gen1_gen2.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen1_gen2.csv

src_dir="generated_images_gen2"
dest_dir="data/generated_images_gen2"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

python train.py  \
  --model_dir weights/weights_gen2\
  --train_csv data/combined_train_gen1_gen2.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2  \
  --arch cgan \
  --epochs 1 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

echo "gen3"
python test.py \
    --num_images 5000 \
    --model-path weights/weights_gen2/GAN-last.pth  \
    --output_csv data/gen3_test.csv  \
    --image_path generated_images_gen3\
    --gen_num 3

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen3\
    --csv_filename data/gen3_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen1_gen2.csv\
    --generated_csv_path data/gen3_test.csv \
    --new_csv_name data/combined_train_gen2_gen3.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen2_gen3.csv

src_dir="generated_images_gen3"
dest_dir="data/generated_images_gen3"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

python train.py  \
  --model_dir weights/weights_gen3\
  --train_csv data/combined_train_gen2_gen3.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3 \
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
    --model-path weights/weights_gen3/GAN-last.pth  \
    --output_csv data/gen4_test.csv  \
    --image_path generated_images_gen4\
    --gen_num 4

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen4\
    --csv_filename data/gen4_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen2_gen3.csv\
    --generated_csv_path data/gen4_test.csv \
    --new_csv_name data/combined_train_gen3_gen4.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen3_gen4.csv

src_dir="generated_images_gen4"
dest_dir="data/generated_images_gen4"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."
 

python train.py  \
  --model_dir weights/weights_gen4\
  --train_csv data/combined_train_gen3_gen4.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4 \
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
    --model-path weights/weights_gen4/GAN-last.pth  \
    --output_csv data/gen5_test.csv  \
    --image_path generated_images_gen5\
    --gen_num 5

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen5\
    --csv_filename data/gen5_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen3_gen4.csv\
    --generated_csv_path data/gen5_test.csv \
    --new_csv_name data/combined_train_gen4_gen5.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen4_gen5.csv

src_dir="generated_images_gen5"
dest_dir="data/generated_images_gen5"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

echo "gen6"
python train.py  \
  --model_dir weights/weights_gen5\
  --train_csv data/combined_train_gen4_gen5.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4,data/generated_images_gen5 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000\
    --model-path weights/weights_gen5/GAN-last.pth  \
    --output_csv data/gen6_test.csv  \
    --image_path generated_images_gen6\
    --gen_num 6

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen6\
    --csv_filename data/gen6_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen4_gen5.csv\
    --generated_csv_path data/gen6_test.csv \
    --new_csv_name data/combined_train_gen5_gen6.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen5_gen6.csv

src_dir="generated_images_gen6"
dest_dir="data/generated_images_gen6"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

echo "gen7"

python train.py  \
  --model_dir weights/weights_gen6\
  --train_csv data/combined_train_gen5_gen6.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4,data/generated_images_gen5,data/generated_images_gen6 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights/weights_gen6/GAN-last.pth  \
    --output_csv data/gen7_test.csv  \
    --image_path generated_images_gen7\
    --gen_num 7

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen7\
    --csv_filename data/gen7_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen5_gen6.csv\
    --generated_csv_path data/gen7_test.csv \
    --new_csv_name data/combined_train_gen6_gen7.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen6_gen7.csv

src_dir="generated_images_gen7"
dest_dir="data/generated_images_gen7"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

echo "gen8"

python train.py  \
  --model_dir weights/weights_gen7\
  --train_csv data/combined_train_gen6_gen7.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4,data/generated_images_gen5,data/generated_images_gen6,data/generated_images_gen7 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights/weights_gen7/GAN-last.pth  \
    --output_csv data/gen8_test.csv  \
    --image_path generated_images_gen8\
    --gen_num 8

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen8\
    --csv_filename data/gen8_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen6_gen7.csv\
    --generated_csv_path data/gen8_test.csv \
    --new_csv_name data/combined_train_gen7_gen8.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen7_gen8.csv

src_dir="generated_images_gen8"
dest_dir="data/generated_images_gen8"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

echo "gen9"

python train.py  \
  --model_dir weights/weights_gen8\
  --train_csv data/combined_train_gen7_gen8.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4,data/generated_images_gen5,data/generated_images_gen6,data/generated_images_gen7,data/generated_images_gen8 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights/weights_gen8/GAN-last.pth  \
    --output_csv data/gen9_test.csv  \
    --image_path generated_images_gen9\
    --gen_num 9

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen9\
    --csv_filename data/gen9_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen7_gen8.csv\
    --generated_csv_path data/gen9_test.csv \
    --new_csv_name data/combined_train_gen8_gen9.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen8_gen9.csv

src_dir="generated_images_gen9"
dest_dir="data/generated_images_gen9"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

echo "gen10"

python train.py  \
  --model_dir weights/weights_gen9\
  --train_csv data/combined_train_gen8_gen9.csv  \
  --image_dirs MNIST/mnist_train,data/generated_images_gen1,data/generated_images_gen2,data/generated_images_gen3,data/generated_images_gen4,data/generated_images_gen5,data/generated_images_gen6,data/generated_images_gen7,data/generated_images_gen8,data/generated_images_gen9 \
  --arch cgan \
  --epochs 50 \
  --start-epoch 0 \
  --b 16 \
  --lr 0.0004 \
  --image-size 28 \
  --channels 3 

python test.py \
    --num_images 8000 \
    --model-path weights/weights_gen9/GAN-last.pth  \
    --output_csv data/gen10_test.csv  \
    --image_path generated_images_gen10\
    --gen_num 10

python detect_color/detect_color.py\
    --base_image_dir generated_images_gen10\
    --csv_filename data/gen10_test.csv
 
python augment/augment_dataset.py \
    --train_csv_filename data/combined_train_gen8_gen9.csv\
    --generated_csv_path data/gen10_test.csv \
    --new_csv_name data/combined_train_gen9_gen10.csv
 
python augment/seedata.py\
    --csv_filename data/combined_train_gen9_gen10.csv

src_dir="generated_images_gen10"
dest_dir="data/generated_images_gen10"
 
mkdir -p "$dest_dir"
 
find "$src_dir" -mindepth 2 -type f -name "*.png" -exec mv {} "$dest_dir" \;

echo "All images have been copied to $dest_dir."
rm -r "$src_dir"   
echo "$src_dir has been removed."

 