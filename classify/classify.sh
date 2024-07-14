#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=gen380.txt  
 

source ~/.bashrc

conda activate llava-med
 
# echo "gen 0"
# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_train \
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen0\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen0 \
 
echo "gen 1"
# echo "gen 1 20%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen1_20per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen1_20per \

# echo "gen 1 50%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen1_50per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen1_50per \

echo "gen 1 70%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen1_70per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen1_70per \

# echo "gen 1 80%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen1_80per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen1_80per \
echo "gen 1 100%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen1_100per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen1_100per \

echo "gen 2 "

# echo "gen 2 20%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen2_20per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen2_20per \

# echo "gen 2 50%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen2_50per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen2_50per \


echo "gen 2 70%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen2_70per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen2_70per \

# echo "gen 2 80%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen2_80per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen2_80per \


echo "gen 2 100%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen2_100per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen2_100per \


echo "gen 3 "

# echo "gen 3 20%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/20percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen3_20per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen3_20per \

# echo "gen 3 50%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/50percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen3_50per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen3_50per \

echo "gen 3 70%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/70percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen3_70per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen3_70per \


# echo "gen 3 80%"

# python classify.py \
#     --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/80percent/train.csv \
#     --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
#     --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/cmnist_train\
#     --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
#     --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen3_80per\
#     --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
#     --model_name resnet50_gen3_80per \

echo "gen 3 100%"

python classify.py \
    --train_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/100percent/train.csv \
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --train_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation3/cmnist_train\
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/results/gen3_100per\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/classify/model \
    --model_name resnet50_gen3_100per \
