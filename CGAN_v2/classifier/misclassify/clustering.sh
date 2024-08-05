#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=clustering
#SBATCH --output=clustering.txt  
 

source ~/.bashrc

conda activate llava-med
 
python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen0 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen0/all_images_results.csv


python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen1 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen1/all_images_results.csv

python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen2 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen2/all_images_results.csv

 
python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen3 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen3/all_images_results.csv














python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen4 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen4/all_images_results.csv
 


python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen5\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen5 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen5/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen5/all_images_results.csv



python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen6\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen6 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen6/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen6/all_images_results.csv
 

python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen7\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen7 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen7/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen7/all_images_results.csv
 

python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen8\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen8 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen8/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen8/all_images_results.csv
 
python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen9\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen9 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen9/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen9/all_images_results.csv
 


python get_misclassified_images.py\
    --test_csv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/test.csv \
    --test_images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --result_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen10\
    --model_save_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/model \
    --model_name resnet50_gen10 \

 
python get_all_results.py \
    --misclassified_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen10/misclassified_images.csv \
    --images_directory /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation0/mnist_test \
    --output_csv_path /local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results/gen10/all_images_results.csv
 