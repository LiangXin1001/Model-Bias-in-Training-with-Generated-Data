import os
import pandas as pd
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate Disparate Impact (DI) for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()

# 设置保存路径
 
base_dir = os.path.dirname(args.result_dir)
save_path = os.path.join(base_dir,'metric_results', args.model_name)
os.makedirs(save_path, exist_ok=True)

# 计算 Disparate Impact 的函数
def calculate_disparate_impact_for_run_id(df, run_id):
    results = []
    superclasses = df['True Superclass'].unique()

    for superclass in superclasses:
        # 获取当前 superclass 下的所有 subclass
        subclasses = df[df['True Superclass'] == superclass]['True Subclass'].unique()
        print("subclasses ",subclasses)
        for subclass in subclasses:
            # 筛选出当前 Run ID 的数据
            subset = df[(df['Run ID'] == run_id) & (df['True Superclass'] == superclass)]
            if subset.empty:
                print("subset   empty ")
                continue
            # 计算当前 subclass 的基础率
            base_rate_current = (subset[subset['True Subclass'] == subclass]['Predicted Superclass'] == superclass).mean()
            # 计算其他 subclass 的基础率
            base_rate_others = (subset[subset['True Subclass'] != subclass]['Predicted Superclass'] == superclass).mean()

            # 计算差异影响，并避免除以零
            if base_rate_others > 0:
                disparate_impact = min(base_rate_current / base_rate_others, base_rate_others / base_rate_current)
            else:
                disparate_impact = None  # 如果基础率为零，则差异影响无法计算
            
            true_superclass_name = subset['True Superclass Name'].iloc[0]
            true_subclass_name = subset[subset['True Subclass'] == subclass]['True Subclass Name'].iloc[0]

            results.append({
                'Run ID': run_id,
                'Superclass': superclass,
                'Subclass': subclass,
                'Disparate Impact': disparate_impact,
                'True Superclass Name': true_superclass_name,
                'True Subclass Name': true_subclass_name
            })

    return results

# 处理每个CSV文件
for i in range(11):
    result_csv = os.path.join(args.result_dir, f'test_results_{i}.csv')
    
    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)

        # 获取所有的 Run IDs
        run_ids = df['Run ID'].unique()
        all_disparate_impact_results = []

        # 对于每个 Run ID 计算 Disparate Impact
        for run_id in run_ids:
            disparate_impact_results = calculate_disparate_impact_for_run_id(df, run_id)
            all_disparate_impact_results.extend(disparate_impact_results)

        # 保存结果到指定路径
        results_df = pd.DataFrame(all_disparate_impact_results)
        di_save_path = os.path.join(save_path, f'disparate_impact_results_{i}.csv')
        results_df.to_csv(di_save_path, index=False)

        # 打印结果
        print(f"Disparate Impact Results for test_results_{i}.csv:")
        print(results_df)
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出

    else:
        print(f"Result CSV {result_csv} does not exist. Skipping.")

print(f"All Disparate Impact results have been saved to {save_path}")