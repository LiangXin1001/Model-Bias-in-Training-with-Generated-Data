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
TEST_MODEL_PY="test_model.py"

# 定义基本模型名
base_model_name="resnet50"

# 循环执行分类和测试任务
for gen in {0..10}
do
    echo "gen $gen"

    # 生成完整的模型名称，包括基本模型名和生成号
    full_model_name="${base_model_name}_gen$gen"

    if [ $gen -eq 0 ]
    then
        train_csv="${BASE_DIR}augment/augmented_train_dataset.csv"
    else
        train_csv="${BASE_DIR}data/combined_train_gen$((gen-1))_gen$gen.csv"
    fi

    # 构建模型保存路径和结果保存路径
    model_save_path="${BASE_DIR}classifier/model/${base_model_name}"
    result_save_path="${BASE_DIR}classifier/results/${base_model_name}"

    # 创建模型保存目录和结果保存目录
    mkdir -p "$model_save_path"
    mkdir -p "$result_save_path"

    # 调用 classify.py
    python $CLASSIFY_PY \
        --train_csv $train_csv \
        --test_csv "${BASE_DIR}MNIST/test.csv" \
        --train_images_dir "${BASE_DIR}MNIST/mnist_train,$(printf "${BASE_DIR}data/generated_images_gen%s," $(seq 1 $gen) | sed 's/,$//')" \
        --test_images_dir "${BASE_DIR}MNIST/mnist_test" \
        --result_save_path "$result_save_path" \
        --model_save_path "$model_save_path" \
        --model_name $full_model_name  

    # 调用 test_model.py
    python $TEST_MODEL_PY \
        --test_csv "${BASE_DIR}MNIST/test.csv" \
        --test_images_dir "${BASE_DIR}MNIST/mnist_test" \
        --result_save_path "$result_save_path" \
        --model_save_path "$model_save_path" \
        --model_name $full_model_name  

done
