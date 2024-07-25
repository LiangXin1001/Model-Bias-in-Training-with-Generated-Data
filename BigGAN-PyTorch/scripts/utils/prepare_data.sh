#!/bin/bash
#SBATCH --output=make_hdf5.out         
#SBATCH --error=make_hdf5.err     
#SBATCH --gres=gpu:2               
#SBATCH --mem=16G        
source ~/.bashrc

conda activate llava-med

python make_hdf5.py \
 --dataset CMNIST \
 --batch_size 256 \
 --data_root data  \
 --train_csv  /local/scratch/hcui25/Project/xin/CS/GAN/BigGAN-PyTorch/data/CMNIST/train.csv\
 --image_dirs /local/scratch/hcui25/Project/xin/CS/GAN/BigGAN-PyTorch/data/CMNIST/mnist_train
python calculate_inception_moments.py --dataset CMNIST --data_root data