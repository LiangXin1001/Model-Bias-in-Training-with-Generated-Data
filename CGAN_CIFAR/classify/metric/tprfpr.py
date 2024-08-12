import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import argparse

# 命令行参数
parser = argparse.ArgumentParser(description='Calculate TPR and FPR for each Subclass in a Superclass')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model (used for saving results)')

args = parser.parse_args()

# 创建保存结果的目录
save_path = os.path.join('metric_results', args.model_name)
os.makedirs(save_path, exist_ok=True)

# 定义函数计算TPR和FPR
def calculate_tpr_fpr(df):
    results = []
    superclasses = df['True Superclass'].unique()
    
    for superclass in superclasses:
        subclasses = df[df['True Superclass'] == superclass]['True Subclass'].unique()
        
        for subclass in subclasses:
            subset = df[(df['True Subclass'] == subclass)]
            
            TP = ((subset['True Superclass'] == subset['Predicted Superclass']) & 
                  (subset['True Subclass'] == subclass)).sum()
            FN = ((subset['True Subclass'] == subclass) & 
                  (subset['True Superclass'] != subset['Predicted Superclass'])).sum()
            FP = ((df['True Subclass'] != subclass) & 
                  (df['Predicted Superclass'] == superclass) & 
                  (df['True Superclass'] != superclass)).sum()
            TN = ((df['True Subclass'] != subclass) & 
                  (df['Predicted Superclass'] != superclass) & 
                  (df['True Superclass'] != superclass)).sum()

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            
            # 获取 True Superclass Name 和 True Subclass Name
            true_superclass_name = df[df['True Superclass'] == superclass]['True Superclass Name'].iloc[0]
            true_subclass_name = df[df['True Subclass'] == subclass]['True Subclass Name'].iloc[0]
            
            results.append({
                'Superclass': superclass,
                'Subclass': subclass,
                'TP': TP,
                'FN': FN,
                'FP': FP,
                'TN': TN,
                'TPR': TPR,
                'FPR': FPR,
                'True Superclass Name': true_superclass_name,
                'True Subclass Name': true_subclass_name 
            })
            
    return results
def difference_in_means_by_subclass(df):
    results = []
    
    superclasses = df['True Superclass'].unique()
    
    for superclass in superclasses:
        subclass_accuracies = []
        subclasses = df[df['True Superclass'] == superclass]['True Subclass'].unique()
        
        for subclass in subclasses:
            subset = df[(df['True Subclass'] == subclass)]
            accuracy = (subset['True Superclass'] == subset['Predicted Superclass']).mean()
            subclass_accuracies.append(accuracy)
        
        if len(subclass_accuracies) > 1:
            max_diff = max(subclass_accuracies) - min(subclass_accuracies)
            results.append({
                'Superclass': superclass,
                'Subclass': subclass,
                'Max Difference': max_diff
            })
    
    return results

# 处理每个CSV文件
for i in range(11):
    csv_file = os.path.join(args.result_dir, f'test_results_{i}.csv')
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        tpr_fpr_results = calculate_tpr_fpr(df)
        subclass_difference_results = difference_in_means_by_subclass(df)
        
        # 保存 TPR 和 FPR 结果到 CSV 文件
        tpr_fpr_df = pd.DataFrame(tpr_fpr_results)
        tpr_fpr_df.to_csv(os.path.join(save_path, f'tpr_fpr_results_{i}.csv'), index=False)
        
        # 保存子类差异结果到 CSV 文件
        subclass_diff_df = pd.DataFrame.from_dict(subclass_difference_results, orient='index')
        subclass_diff_df.to_csv(os.path.join(save_path, f'subclass_difference_results_{i}.csv'))

        # 打印 Difference in Means
        print(f"Difference in Means for test_results_{i}:")
        for key, value in subclass_difference_results.items():
            print(f"{key}: {value}")
        
        # 打印 TPR 和 FPR
        print(f"\nTPR and FPR for test_results_{i}:")
        for index, row in tpr_fpr_df.iterrows():
            print(f"Superclass {row['Superclass']} ({row['True Superclass Name']}), Subclass {row['Subclass']} ({row['True Subclass Name']}): TPR = {row['TPR']}, FPR = {row['FPR']}")
        
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出
    
    else:
        print(f"CSV file {csv_file} does not exist. Skipping.")

print(f"Results have been saved to {save_path}")
