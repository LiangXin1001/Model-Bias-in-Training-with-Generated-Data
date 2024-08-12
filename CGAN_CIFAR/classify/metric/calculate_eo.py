import pandas as pd
import argparse
import os
 
# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate EO for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()

# 设置保存路径
result_dir = args.result_dir
save_path = os.path.join('metric_results', args.model_name)
os.makedirs(save_path, exist_ok=True)

# 定义获取TPR的函数
def get_tpr(tprfpr_df, superclass, subclass):
    row = tprfpr_df[(tprfpr_df['Superclass'] == superclass) & (tprfpr_df['Subclass'] == subclass)]
    return row['TPR'].values[0] if not row.empty else None

# 处理每个CSV文件
for i in range(11):
    result_csv = os.path.join(result_dir, f'test_results_{i}.csv')
    tprfpr_csv = os.path.join(save_path, f'tpr_fpr_results_{i}.csv')
    
    if os.path.exists(result_csv) and os.path.exists(tprfpr_csv):
        results_df = pd.read_csv(result_csv)
        tprfpr_df = pd.read_csv(tprfpr_csv)

        # 计算每个Superclass的EO
        eo_values = {}
        superclasses = results_df['True Superclass'].unique()

        for superclass in superclasses:
            subclasses = results_df[results_df['True Superclass'] == superclass]['True Subclass'].unique()
            tpr_values = [get_tpr(tprfpr_df, superclass, subclass) for subclass in subclasses]
            
            # 获取 True Superclass Name 和 True Subclass Name
            true_superclass_name = results_df[results_df['True Superclass'] == superclass]['True Superclass Name'].iloc[0]
            
            for subclass in subclasses:
                true_subclass_name = results_df[results_df['True Subclass'] == subclass]['True Subclass Name'].iloc[0]
            
                # 确保所有TPR值都存在
                if None not in tpr_values and len(tpr_values) > 1:
                    # 计算各组之间的差异
                    diff_sum = sum(abs(tpr_values[i] - tpr_values[j]) for i in range(len(tpr_values)) for j in range(i + 1, len(tpr_values)))
                    
                    # 计算平均差异
                    avg_diff = diff_sum / (len(tpr_values) * (len(tpr_values) - 1) / 2)
                    
                    # 计算EO
                    eo = 1 - avg_diff
                    eo_values[(superclass, subclass)] = {
                        'EO': eo,
                        'True Superclass Name': true_superclass_name,
                        'True Subclass Name': true_subclass_name
                    }

        # 生成保存路径
        eo_list = [{'Superclass': superclass, 
                    'Subclass': subclass,
                    'EO': eo_info['EO'],
                     'True Superclass Name': eo_info['True Superclass Name'],
                    'True Subclass Name': eo_info['True Subclass Name']
                    } for (superclass, subclass), eo_info in eo_values.items()]
        
        eo_df = pd.DataFrame(eo_list)
        eo_save_path = os.path.join(save_path, f'eo_results_{i}.csv')
        eo_df.to_csv(eo_save_path, index=False)

        # 打印结果
        print(f"EO Results for test_results_{i}.csv:")
        for (superclass, subclass), eo_info in eo_values.items():
            print(f"Superclass {superclass} ({eo_info['True Superclass Name']}), Subclass {subclass} ({eo_info['True Subclass Name']}): EO = {eo_info['EO']}")
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出

    else:
        print(f"Result CSV {result_csv} or TPR/FPR CSV {tprfpr_csv} does not exist. Skipping.")

print(f"All EO results have been saved to {save_path}")
