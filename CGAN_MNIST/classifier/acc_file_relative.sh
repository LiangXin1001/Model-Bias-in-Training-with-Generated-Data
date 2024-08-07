#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=clustering
#SBATCH --output=clustering.txt  

source ~/.bashrc
conda activate llava-med

# 定义基础路径
BASE_DIR="../"

# 定义公共路径
TEST_CSV="${BASE_DIR}MNIST/test.csv"
TEST_IMAGES_DIR="${BASE_DIR}MNIST/mnist_test"

# 执行分类和结果统计
for gen in {0..10}
do
    RESULT_SAVE_PATH="${BASE_DIR}classifier/results/gen${gen}"
    MODEL_SAVE_PATH="${BASE_DIR}classifier/model"
    MODEL_NAME="resnet50_gen${gen}"
    MISCLASSIFIED_CSV_PATH="${RESULT_SAVE_PATH}/misclassified_images.csv"
    ALL_IMAGES_RESULTS_CSV="${RESULT_SAVE_PATH}/all_images_results.csv"

    # 获取误分类的图像
    python misclassify/get_misclassified_images.py \
        --test_csv $TEST_CSV \
        --test_images_dir $TEST_IMAGES_DIR \
        --result_save_path $RESULT_SAVE_PATH \
        --model_save_path $MODEL_SAVE_PATH \
        --model_name $MODEL_NAME

    # 获取所有结果
    python misclassify/get_all_results.py \
        --misclassified_csv_path $MISCLASSIFIED_CSV_PATH \
        --images_directory $TEST_IMAGES_DIR \
        --output_csv_path $ALL_IMAGES_RESULTS_CSV

done
