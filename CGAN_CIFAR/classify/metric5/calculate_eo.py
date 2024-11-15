import pandas as pd
import argparse
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description='Calculate ovrEO for each Superclass')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')
 
args = parser.parse_args()

 
base_dir = os.path.dirname(args.result_dir)
save_path = os.path.join(base_dir,'metric_results', args.model_name)
 
os.makedirs(save_path, exist_ok=True)

# 定义获取TPR的函数
 
def get_tpr(tprfpr_df, superclass,subclass, run_id):
    # row = tprfpr_df[(tprfpr_df['Superclass'] == superclass) & (tprfpr_df['Run ID'] == run_id)]
    row = tprfpr_df[(tprfpr_df['Superclass'].astype(int) == int(superclass)) & 
                (tprfpr_df['Run ID'].astype(int) == int(run_id)) & 
                (tprfpr_df['Subclass'].astype(int) == int(subclass))]
    print(" row['TPR'].values ", row['TPR'].values )
    return row['TPR'].values[0] if not row.empty else None
 
 
# 处理每个CSV文件


for i in range(11):
    result_csv = os.path.join(args.result_dir, f'test_results_{i}.csv')
    tprfpr_csv = os.path.join(save_path, f'tpr_fpr_results_{i}.csv')
    all_eo_results = []
    if os.path.exists(result_csv) and os.path.exists(tprfpr_csv):
        results_df = pd.read_csv(result_csv, header=0)
        tprfpr_df = pd.read_csv(tprfpr_csv, header=0)

        # 检查并删除表头重复行，防止它们作为数据的一部分
        if 'Image' in results_df.columns:
            results_df = results_df[results_df['Image'] != 'Image']
        
        # 重置索引并继续
        results_df = results_df.reset_index(drop=True)
        # Run ID,Superclass,Subclass,TP,FN,FP,TN,TPR,FPR,
        numeric_columns = ['Run ID', 'Superclass', 'Subclass', 'TP', 'FN','FP','TN','TPR','FPR']
    
        tprfpr_df[numeric_columns] = tprfpr_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
     
    
        # 获取所有的 Run IDs
        run_ids = results_df['Run ID'].unique()

        # 对于每个 Run ID 计算每个Superclass的ovrEO
        for run_id in run_ids:
             
            eo_values = {}
            results_run_df = results_df[results_df['Run ID'] == run_id]
            superclasses = results_run_df['True Superclass'].unique()
            # print('superclasses',superclasses)
            # all_tpr = [get_tpr(tprfpr_df, sc, run_id) for sc in superclasses]
            # print(f"All TPR values for test_results_{i}.csv: {all_tpr}")
 
            

            for superclass in superclasses:
                subclasses = results_run_df[results_run_df['True Superclass'] == superclass]['True Subclass'].unique()
                print("subclasses",subclasses)
                tpr_values = [get_tpr(tprfpr_df, superclass, subclass,run_id) for subclass in subclasses]
                print("tpr_values",tpr_values)
                # 获取 True Superclass Name 和 True Subclass Name
                true_superclass_name = results_run_df[results_run_df['True Superclass'] == superclass]['True Superclass Name'].iloc[0]
                
                for subclass in subclasses:
                    true_subclass_name = results_run_df[results_run_df['True Subclass'] == subclass]['True Subclass Name'].iloc[0]
                
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

          
            # 将 Run ID 和 ovrEO 结果添加到 all_eo_results
            for (superclass, subclass), eo_info in eo_values.items():
                all_eo_results.append({
                    'Run ID': run_id,
                    'Superclass': superclass,
                    'Subclass': subclass,
                    'EO': eo_info['EO'],
                    'True Superclass Name': eo_info['True Superclass Name'],
                    'True Subclass Name': eo_info['True Subclass Name'] 
                })

        # 保存每个 i 的 ovrEO 结果到一个文件
        eo_df = pd.DataFrame(all_eo_results)
        eo_save_path = os.path.join(save_path, f'eo_results_{i}.csv')
        eo_df.to_csv(eo_save_path, index=False)
 

        # 打印结果
        print(f"ovrEO Results for test_results_{i}.csv:")
        for superclass, eo_info in eo_values.items():
            print(f"Run ID {run_id}, Superclass {superclass} ({eo_info['True Superclass Name']}): ovrEO = {eo_info['EO']:.4f}")
        print("\n" + "="*50 + "\n")  # 分割每个文件的输出

    else:
        print(f"Result CSV {result_csv} or TPR/FPR CSV {tprfpr_csv} does not exist. Skipping.")

# # 保存所有的 ovrEO 结果到一个文件
# eo_df = pd.DataFrame(all_eo_results)
# eo_save_path = os.path.join(save_path, f'eo_ovr_results_all.csv')
# eo_df.to_csv(eo_save_path, index=False)
 
# print(f"All ovrEO results have been saved to {eo_save_path}")
 
