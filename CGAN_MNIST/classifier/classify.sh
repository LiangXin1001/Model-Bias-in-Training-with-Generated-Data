#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=gen380.txt  
 

source ~/.bashrc

conda activate llava-med
 
# echo "gen 0"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/augmented_train_dataset.csv\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen0 \
 
# echo "gen 1"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen0_gen1.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1  \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen1 \

# echo "gen 2"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen1_gen2.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3 \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen2 \

# echo "gen 3"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen2_gen3.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3 \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen3 \




# echo "gen 4"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen3_gen4.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen4 \
  

# echo "gen 5"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen4_gen5.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5  \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen5\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen5 \



# echo "gen 6"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen5_gen6.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6   \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen6\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen6  

# echo "gen 7"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen6_gen7.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7   \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen7\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen7  

# echo "gen 8"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen7_gen8.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8   \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen8\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen8  
# echo "gen 9"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen8_gen9.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen9   \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen9\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen9  

# echo "gen 10"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen9_gen10.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN_v2/mnist_train,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen1,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen2,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen3,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen4,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen6,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen5,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen7,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen8,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen9,/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/generated_images_gen10   \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen10\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen10 



# python test_model.py\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen0 \


# python test_model.py\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen1 \

# python test_model.py\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen2 \
 

# python test_model.py\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen3 \
 

# python test_model.py\
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
#     --model_name resnet50_gen4 \




echo "gen 4"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen4 
  

echo "gen 5"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen5\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen5 



echo "gen 6"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen6\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen6    

echo "gen 7"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen7\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen7 

echo "gen 8"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen8\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen8 
echo "gen 9"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen9\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen9 

echo "gen 10"
python test_model.py \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen10\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen10 
