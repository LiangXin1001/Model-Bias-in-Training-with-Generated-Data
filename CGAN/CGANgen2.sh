#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=C
#SBATCH --output=C.txt  

source ~/.bashrc

conda activate llava-med
 
  
 

echo "generation 2 "

# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent/train.csv \
#    --genmodel "second_generation_20percent_gen.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/synthetizedtraincsv.csv \
#    --gen_num 2 \
   
 

# echo "generation 2 20%"
 
# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/20percent/train.csv  \
#    --genmodel "second_generation_20percent_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/top20synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/train.csv \
#    --toppercent 0.2



# echo "generation 2 50%"


# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent/train.csv \
#    --genmodel "second_generation_50percent_gen.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/synthetizedtraincsv.csv \
#    --gen_num 2 \
   
 

# echo "generation 2 50%"
 
# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/50percent/train.csv  \
#    --genmodel "second_generation_50percent_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/top50synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/train.csv \
#    --toppercent 0.5



echo "generation 2 70%"


python generate_images_test.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent/train.csv \
   --genmodel "second_generation_70percent_gen.pt" \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/synthetizedtraincsv.csv \
   --gen_num 2 \
   
 

echo "generation 2 70%"
 
python make_next_gen_data.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/70percent/train.csv  \
   --genmodel "second_generation_70percent_gen.pt" \
   --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/synthetizedtraincsv.csv \
   --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/top70synthetizedcsv.csv \
   --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/train.csv \
   --toppercent 0.7


# echo "generation 2 80%"


# python generate_images_test.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent/train.csv \
#    --genmodel "second_generation_80percent_gen.pt" \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/synthetizedtraincsv.csv \
#    --gen_num 2 \
   
 

# echo "generation 1 80%"
 
# python make_next_gen_data.py \
#    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#    --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/80percent/train.csv  \
#    --genmodel "second_generation_80percent_gen.pt" \
#    --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent \
#    --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/synthetizedtraincsv.csv \
#    --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/top80synthetizedcsv.csv \
#    --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/train.csv \
#    --toppercent 0.8



echo "generation 2 100%"


python generate_images_test.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent/train.csv \
   --genmodel "second_generation_100percent_gen.pt" \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/synthetizedtraincsv.csv \
   --gen_num 2 \
   
 

echo "generation 2 100%"
 
python make_next_gen_data.py \
   --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
   --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
   --traincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/100percent/train.csv  \
   --genmodel "second_generation_100percent_gen.pt" \
   --percent_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent \
   --synthetizedtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/synthetizedtraincsv.csv \
   --top10synthetizedcsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/top100synthetizedcsv.csv \
   --newtraincsv /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/train.csv \
   --toppercent 1

 echo "clear images "
# python clearimages.py \
#    --folder_a  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train  \
#    --folder_b /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train  \
#    --csv_file /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent/train.csv
 
 
# python clearimages.py \
#    --folder_a  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train  \
#    --folder_b /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train  \
#    --csv_file /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent/train.csv

echo "70 % "
python clearimages.py \
   --folder_a  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train  \
   --folder_b /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train  \
   --csv_file /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent/train.csv

# echo "80 % "
# python clearimages.py \
#    --folder_a  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train  \
#    --folder_b /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train  \
#    --csv_file /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent/train.csv
echo "100 % "
python clearimages.py \
   --folder_a  /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train  \
   --folder_b /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation1/cmnist_train  \
   --csv_file /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent/train.csv

echo "clear done"

# python CGAN.py \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/20percent \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --genmodel "third_generation_20percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \

# python CGAN.py \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/50percent \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#     --genmodel "third_generation_50percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \


python CGAN.py \
    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
    --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/70percent \
    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
    --genmodel "third_generation_70percent_gen.pt" \
    --num_epochs 50 \
    --n_critic 5 \
    --n_generator 5 \
    --display_step 100 \


# python CGAN.py \
#     --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
#     --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/80percent \
#     --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
#     --genmodel "third_generation_80percent_gen.pt" \
#     --num_epochs 50 \
#     --n_critic 5 \
#     --n_generator 5 \
#     --display_step 100 \


python CGAN.py \
    --images_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/cmnist_train \
    --base_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/generation2/100percent \
    --model_dir /local/scratch/hcui25/Project/xin/CS/GAN/CGAN/model \
    --genmodel "third_generation_100percent_gen.pt" \
    --num_epochs 50 \
    --n_critic 5 \
    --n_generator 5 \
    --display_step 100 \


