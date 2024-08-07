#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=metric
#SBATCH --output=metric.txt  

source ~/.bashrc
conda activate llava-med

# 定义基础路径，从当前目录指向 metrics_analysize 子目录
SCRIPTS_DIR="./metrics_analysize/"

# 循环处理每一个generation
for gen in {0..10}
do
    echo "gen${gen}"

    RESULT_CSV="results/gen${gen}/all_images_results.csv"
    RESULT_SAVE_PATH="results/gen${gen}"
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
