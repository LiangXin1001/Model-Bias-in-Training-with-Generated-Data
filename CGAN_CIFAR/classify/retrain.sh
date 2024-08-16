#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=retrain.txt  
#SBATCH --error=retrain.err     
# 设置基本路径变量，向上返回到 CGAN-PyTorch 目录
BASE_DIR="/local/scratch/hcui25/Project/xin/CS/GAN/CGAN_CIFAR/"

# 激活环境
source ~/.bashrc
conda activate llava-med

# python retrain.py


# 定义 classify.py 和 test_model.py 的相对路径
CLASSIFY_PY="retrain.py"
 
# 定义基本模型名
base_model_name="resnet50"
 
python $CLASSIFY_PY \
    --gennum 0 \
    --model_name $base_model_name

python test.py \
    --gennum 0 \
    --model_name $base_model_name

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
    # python $CLASSIFY_PY \
    #     --gennum $gennum \
    #     --data_root_paths "$data_root_paths"\
    #     --model_name $base_model_name
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name
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
    # python $CLASSIFY_PY \
    #     --gennum $gennum \
    #     --data_root_paths "$data_root_paths"\
    #     --model_name $base_model_name
    python test.py \
        --gennum $gennum \
        --model_name $base_model_name
done

echo "All runs completed."