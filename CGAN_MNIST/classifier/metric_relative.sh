#!/bin/bash
 
conda init
. ~/.bashrc
conda init bash

conda activate xin

# 定义基础路径，从当前目录指向 metrics_analysize 子目录
SCRIPTS_DIR="./metrics_analysize/"
echo "Current working directory: $(pwd)"

# 循环处理每一个generation
base_model_name="vgg19"
for gen in {0..10}
do
    echo "gen${gen}"

    RESULT_CSV="results/${base_model_name}/gen${gen}/all_images_results.csv"
    RESULT_SAVE_PATH="results/${base_model_name}/gen${gen}"
    TPRFPR_CSV="${RESULT_SAVE_PATH}/tpr_fpr_results.csv"

    # 执行 tprfpr.py 分析
    python ${SCRIPTS_DIR}tprfpr.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 residuals.py 分析
    python ${SCRIPTS_DIR}residuals.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 disparate_impact.py 分析
    python ${SCRIPTS_DIR}disparate_impact.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 EO.py 分析
    python ${SCRIPTS_DIR}EO.py \
        --result_csv $RESULT_CSV \
        --tprfpr_csv $TPRFPR_CSV
done


base_model_name="alexnet"
for gen in {0..10}
do
    echo "gen${gen}"

    RESULT_CSV="results/${base_model_name}/gen${gen}/all_images_results.csv"
    RESULT_SAVE_PATH="results/${base_model_name}/gen${gen}"
    TPRFPR_CSV="${RESULT_SAVE_PATH}/tpr_fpr_results.csv"

    # 执行 tprfpr.py 分析
    python ${SCRIPTS_DIR}tprfpr.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 residuals.py 分析
    python ${SCRIPTS_DIR}residuals.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 disparate_impact.py 分析
    python ${SCRIPTS_DIR}disparate_impact.py \
        --result_csv $RESULT_CSV \
        --result_save_path $RESULT_SAVE_PATH

    # 执行 EO.py 分析
    python ${SCRIPTS_DIR}EO.py \
        --result_csv $RESULT_CSV \
        --tprfpr_csv $TPRFPR_CSV
done