#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=class.txt  

# 设置基本路径变量，向上返回到 CGAN-PyTorch 目录
BASE_DIR="../"

# 激活环境
source ~/.bashrc
conda activate llava-med

# 定义 classify.py 和 test_model.py 的相对路径
CLASSIFY_PY="classify.py"
 
# 定义基本模型名
base_model_name="resnet50"
 
for gennum in {0..9}
do
    echo "Running with gennum = $gennum"
    python $CLASSIFY_PY \
        --gennum $gennum \
        --data_root_paths ../data/generated_images_{args.gennum} \
        --model_name $base_model_name

#TODO
    # python test.py \
    #     --gennum $gennum \
    #     --model_name $base_model_name
done

echo "All runs completed."


