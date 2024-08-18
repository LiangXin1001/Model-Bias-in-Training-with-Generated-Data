#!/bin/bash

gennum=0
mkdir -p logs
python mainACGAN.py \
    --gennum $gennum >> logs/training.txt 2>&1

python gendata.py \
    --gennum $gennum  \
    --model_path models >> logs/training.txt 2>&1

for gennum in {1..6}
do
    start_idx=0
    end_idx=$((gennum - 1))

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
    echo "Generated data_root_paths: $data_root_paths" >> logs/training.txt 2>&1
    python mainACGAN.py \
        --gennum ${gennum} \
        --data_root_paths "$data_root_paths" >> logs/training.txt 2>&1

    python gendata.py \
        --gennum $gennum  \
        --model_path models >> logs/training.txt 2>&1

done

gennum=7

for gennum in {7..10}
do
    start_idx=$((gennum - 4))
    end_idx=$((gennum - 1))

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
    echo "Generated data_root_paths: $data_root_paths" >> logs/training.txt 2>&1
    python mainACGAN.py \
        --gennum ${gennum} \
        --data_root_paths "$data_root_paths" >> logs/training.txt 2>&1

    python gendata.py \
        --gennum $gennum  \
        --model_path models >> logs/training.txt 2>&1

done
