#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=class
#SBATCH --output=gen380.txt  

# 设置基本路径变量，向上返回到 CGAN-PyTorch 目录
BASE_DIR="../"

# 激活环境
source ~/.bashrc
conda activate llava-med

# 定义 classify.py 和 test_model.py 的相对路径
CLASSIFY_PY="classify.py"
TEST_MODEL_PY="test_model.py"

# 循环执行分类和测试任务
for gen in {0..10}
do
    echo "gen $gen"

    if [ $gen -eq 0 ]
    then
        train_csv="${BASE_DIR}augment/augmented_train_dataset.csv"
    else
        train_csv="${BASE_DIR}data/combined_train_gen$((gen-1))_gen$gen.csv"
    fi

    # 调用 classify.py
    python $CLASSIFY_PY \
        --train_csv $train_csv \
        --test_csv "${BASE_DIR}MNIST/test.csv" \
        --train_images_dir "${BASE_DIR}MNIST/mnist_train,$(printf "${BASE_DIR}data/generated_images_gen%s," $(seq 1 $gen) | sed 's/,$//')" \
        --test_images_dir "${BASE_DIR}MNIST/mnist_test" \
        --result_save_path "${BASE_DIR}classifier/results/gen$gen" \
        --model_save_path "${BASE_DIR}classifier/model" \
        --model_name "resnet50_gen$gen"

    # 调用 test_model.py
    python $TEST_MODEL_PY \
        --test_csv "${BASE_DIR}MNIST/test.csv" \
        --test_images_dir "${BASE_DIR}MNIST/mnist_test" \
        --result_save_path "${BASE_DIR}classifier/results/gen$gen" \
        --model_save_path "${BASE_DIR}classifier/model" \
        --model_name "resnet50_gen$gen"

done
