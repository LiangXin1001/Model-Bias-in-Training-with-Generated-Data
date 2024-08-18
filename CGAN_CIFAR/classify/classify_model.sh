#!/bin/bash
 
BASE_DIR="../"
 
 
CLASSIFY_PY="classify.py"
 
# 定义基本模型名
base_model_name="mobilenetv3 "
 
python $CLASSIFY_PY \
    --gennum 0 \
    --model_name $base_model_name \
    --start_train_epoch 0 \
    --epochs 40

python test.py \
    --gennum 0 \
    --model_name $base_model_name \
    --epochs 40

for gennum in {1..6}
do
    start_idx=0
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done


for gennum in {7..10}
do
    start_idx=$((gennum - 4)) 
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done

echo "All runs completed."


# 定义基本模型名
base_model_name="vgg19"
 
python $CLASSIFY_PY \
    --gennum 0 \
    --model_name $base_model_name \
    --start_train_epoch 0 \
    --epochs 40

python test.py \
    --gennum 0 \
    --model_name $base_model_name \
    --epochs 40

for gennum in {1..6}
do
    start_idx=0
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done


for gennum in {7..10}
do
    start_idx=$((gennum - 4)) 
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done

echo "All runs completed."

# 定义基本模型名
base_model_name="resnet50"
 
python $CLASSIFY_PY \
    --gennum 0 \
    --model_name $base_model_name \
    --start_train_epoch 0 \
    --epochs 40

python test.py \
    --gennum 0 \
    --model_name $base_model_name \
    --epochs 40

for gennum in {1..6}
do
    start_idx=0
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done


for gennum in {7..10}
do
    start_idx=$((gennum - 4)) 
    end_idx=$((gennum - 1)) 

    # 构建 data_root_paths 参数
    data_root_paths=""
    for (( i=$start_idx; i<=$end_idx; i++ ))
    do
        if [ "$i" -eq $start_idx ]; then
            data_root_paths="${BASE_DIR}data/generated_images_$i"
        else
            data_root_paths="$data_root_paths,${BASE_DIR}data/generated_images_$i"
        fi
    done
    echo "Generated data_root_paths: $data_root_paths"
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths "$data_root_paths"\
        --model_name $base_model_name \
        --epochs 40
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name\
        --epochs 40
done

echo "All runs completed."
