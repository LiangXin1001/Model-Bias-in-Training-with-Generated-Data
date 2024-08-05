import pandas as pd
import argparse
import os
 
# Set up command line arguments
parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--result_csv', type=str, required=True, help='CSV file path for testing dataset')
parser.add_argument('--tprfpr_csv', type=str, required=True, help='CSV file path for testing dataset')
 
args = parser.parse_args()
 

results_df = pd.read_csv(args.result_csv)
tprfpr_df = pd.read_csv(args.tprfpr_csv)

 
def get_tpr(tprfpr_df, digit, color):
    row = tprfpr_df[(tprfpr_df['Digit'] == digit) & (tprfpr_df['Color'] == color)]
    return row['FPR'].values[0] if not row.empty else None

# 计算每个数字的EO
eo_values = {}
for digit in range(10):
    tpr_values = [get_tpr(tprfpr_df, digit, color) for color in range(3)]
    
    # 确保所有TPR值都存在
    if None not in tpr_values:
        # 计算各组之间的差异
        diff_01 = abs(tpr_values[0] - tpr_values[1])
        diff_02 = abs(tpr_values[0] - tpr_values[2])
        diff_12 = abs(tpr_values[1] - tpr_values[2])
        
        # 计算平均差异
        avg_diff = (diff_01 + diff_02 + diff_12) / 3
        
        # 计算EO
        eo = 1 - avg_diff
        eo_values[digit] = eo
 
directory = os.path.dirname(args.result_csv)

# 生成保存路径
save_path = os.path.join(directory, 'eo_results.csv')

eo_list = [{'Digit': digit, 'EO': eo} for digit, eo in eo_values.items()]
eo_df = pd.DataFrame(eo_list)
eo_df.to_csv(save_path, index=False)

for digit, eo in eo_values.items():
    print(f"Digit {digit}: EO = {eo}")
