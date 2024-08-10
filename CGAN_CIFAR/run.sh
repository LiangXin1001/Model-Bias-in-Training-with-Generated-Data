#!/bin/bash
#SBATCH --job-name=training             
#SBATCH --gres=gpu:2                 
#SBATCH --mem=500G                   
#SBATCH --output=gendata.out       
#SBATCH --error=gendata.err        

source ~/.bashrc
 
conda activate llava-med
export CUDA_LAUNCH_BLOCKING=1

gennum=0

# python mainACGAN.py \
#     --gennum $gennum 

# python gendata.py \
#     --gennum $gennum  \
#     --model_path  models 




for gennum in {1..10}
do
    start_idx=0
    end_idx=$gennum   

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    python mainACGAN.py \
        --gennum ${gennum} \
        --data_root_paths "$data_root_paths"
    python gendata.py \
        --gennum $gennum  \
        --model_path  models 
 
done
