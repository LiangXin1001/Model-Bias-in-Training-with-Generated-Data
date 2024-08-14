import pandas as pd
import argparse
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate EO for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()
 
result_dir = args.result_dir
  
save_path = os.path.join('metric_results', args.model_name)
os.makedirs(save_path, exist_ok=True)
# 计算EO
def calculate_eo(df):
    results = []
    superclasses = df['True Superclass'].unique()
    
    for superclass in superclasses:
        subclasses = df[df['True Superclass'] == superclass]['True Subclass'].unique()
        
        for privileged_subclass in subclasses:
            privileged_subset = df[(df['True Subclass'] == privileged_subclass)]
            TP_priv = ((privileged_subset['Predicted Superclass'] == superclass) & (privileged_subset['True Superclass'] == superclass)).sum()
            FN_priv = ((privileged_subset['True Subclass'] == privileged_subclass) & (privileged_subset['Predicted Superclass'] != superclass)).sum()
            TPRz1 = TP_priv / (TP_priv + FN_priv) if (TP_priv + FN_priv) > 0 else 0
            
            unprivileged_subsets = df[(df['True Superclass'] == superclass) & (df['True Subclass'] != privileged_subclass)]
            TP_unpriv = ((unprivileged_subsets['Predicted Superclass'] == superclass) & (unprivileged_subsets['True Superclass'] == superclass)).sum()
            FN_unpriv = ((unprivileged_subsets['True Subclass'] != privileged_subclass) & (unprivileged_subsets['Predicted Superclass'] != superclass)).sum()
            TPRz0 = TP_unpriv / (TP_unpriv + FN_unpriv) if (TP_unpriv + FN_unpriv) > 0 else 0
            
            EO = 1 - abs(TPRz1 - TPRz0)
            true_superclass_name = df[df['True Superclass'] == superclass]['True Superclass Name'].iloc[0]
            true_subclass_name = df[df['True Subclass'] == privileged_subclass]['True Subclass Name'].iloc[0]
            
            results.append({
                'Superclass': superclass,
                'Privileged Subclass': privileged_subclass,
                'True Superclass Name': true_superclass_name,
                'True Subclass Name': true_subclass_name,
                'TPRz1': TPRz1,
                'TPRz0': TPRz0,
                'EO': EO
            })
    
    return pd.DataFrame(results)


# 处理每个CSV文件
for i in range(11):
    result_csv = os.path.join(result_dir, f'test_results_{i}.csv')
    
    if os.path.exists(result_csv):
        results_df = pd.read_csv(result_csv)
        eo_results = calculate_eo(results_df)
        eo_save_path = os.path.join(save_path, f'eo_twoclass_results_{i}.csv')
        eo_results.to_csv(eo_save_path, index=False)

        # 打印结果
        print(f"EO Results for test_results_{i}.csv:")
        for index, row in eo_results.iterrows():
            print(f"Superclass {row['Superclass']} ({row['True Superclass Name']}), Privileged Subclass {row['Privileged Subclass']} ({row['True Subclass Name']}): EO = {row['EO']}")
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出
    else:
        print(f"Result CSV {result_csv} does not exist. Skipping.")

print(f"All EO results have been saved to {save_path}")

# 打印结果
print(f"EO Results have been saved to {eo_save_path}")
