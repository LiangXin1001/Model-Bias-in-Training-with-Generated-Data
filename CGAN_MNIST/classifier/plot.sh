#!/bin/bash

#!/bin/bash
 
OUTPUT_FILE="plottu.txt"
SCRIPTS_DIR="./plottu/"

# 清空或创建一个新的输出文件
: > $OUTPUT_FILE
  
echo "all figures" >> $OUTPUT_FILE
python plottu/plotedi_for_multiplemodels.py --model_names "vgg19" "resnet50"  "alexnet" >> $OUTPUT_FILE 2>&1
python plottu/acc_for_multiplemodels.py --model_names "vgg19" "resnet50"  "alexnet" >> $OUTPUT_FILE 2>&1
python plottu/ploteo_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet" >> $OUTPUT_FILE 2>&1
python plottu/plottpr_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet" >> $OUTPUT_FILE 2>&1
python plottu/plot_subclass_diff_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet" >> $OUTPUT_FILE 2>&1
 
base_model_name="alexnet"

python ${SCRIPTS_DIR}color_accuracies.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1
python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1
python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1


base_model_name="resnet50"

python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1
python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1


base_model_name="vgg19"

python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1
python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1

python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name} >> $OUTPUT_FILE 2>&1


# SCRIPTS_DIR="./plottu/"
  
# echo "all figures"
# python plottu/plotedi_for_multiplemodels.py --model_names "vgg19" "resnet50"  "alexnet"
# python plottu/acc_for_multiplemodels.py --model_names "vgg19" "resnet50"  "alexnet"
# python plottu/ploteo_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet"
# python plottu/plottpr_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet"
# python plottu/plot_subclass_diff_for_multiplemodels.py --model_names "vgg19" "resnet50" "alexnet"
 
# base_model_name="alexnet"

# python ${SCRIPTS_DIR}color_accuracies.py --model_name ${base_model_name}
# python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name}
# python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name}


# base_model_name="resnet50"

# python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name}
# python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name}



# base_model_name="vgg19"

# python ${SCRIPTS_DIR}average_acc.py --model_name ${base_model_name}
# python ${SCRIPTS_DIR}ploteo.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plot_digit_color_diff.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plotdi.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}plottpr.py --model_name ${base_model_name}

# python ${SCRIPTS_DIR}digit_difference.py --model_name ${base_model_name}