import os
import pandas as pd
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate Disparate Impact (DI) for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()

# 设置保存路径
result_dir = args.result_dir
save_path = os.path.join('metric_results', args.model_name)
os.makedirs(save_path, exist_ok=True)

# 计算 Disparate Impact 的函数
def calculate_disparate_impact_for_all(df):
    results = []
    superclasses = df['True Superclass'].unique()

    for superclass in superclasses:
        # 获取当前 superclass 下的所有 subclass
        subclasses = df[df['True Superclass'] == superclass]['True Subclass'].unique()
        
        for subclass in subclasses:
            # 计算当前 subclass 的基础率
            base_rate_current = (df[(df['True Superclass'] == superclass) & (df['True Subclass'] == subclass)]['Predicted Superclass'] == superclass).mean()
            # 计算其他 subclass 的基础率
            base_rate_others = (df[(df['True Superclass'] == superclass) & (df['True Subclass'] != subclass)]['Predicted Superclass'] == superclass).mean()

            # 计算差异影响，并避免除以零
            if base_rate_others > 0:
                disparate_impact = min(base_rate_current / base_rate_others, base_rate_others / base_rate_current)
            else:
                disparate_impact = None  # 如果基础率为零，则差异影响无法计算
            
            true_superclass_name = df[df['True Superclass'] == superclass]['True Superclass Name'].iloc[0]
            true_subclass_name = df[df['True Subclass'] == subclass]['True Subclass Name'].iloc[0]

            results.append({
                'Superclass': superclass,
                'Subclass': subclass,
                'Disparate Impact': disparate_impact,
                'True Superclass Name': true_superclass_name,
                'True Subclass Name': true_subclass_name 
            })

    return results

# 处理每个CSV文件
for i in range(11):
    result_csv = os.path.join(result_dir, f'test_results_{i}.csv')
    
    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
        
        results = calculate_disparate_impact_for_all(df)
        results_df = pd.DataFrame(results)
        
        # 保存结果到指定路径
        results_df.to_csv(os.path.join(save_path, f'disparate_impact_results_{i}.csv'), index=False)

        # 打印结果
        print(f"Disparate Impact Results for test_results_{i}.csv:")
        print(results_df)
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出

    else:
        print(f"Result CSV {result_csv} does not exist. Skipping.")

print(f"All Disparate Impact results have been saved to {save_path}")
