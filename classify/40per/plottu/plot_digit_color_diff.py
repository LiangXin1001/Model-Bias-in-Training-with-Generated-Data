import pandas as pd
import matplotlib.pyplot as plt
import os

 
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'

 
subdirectories = ['gen0', 'gen1', 'gen2','gen3','gen4']
 
def read_color_diff_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'color_difference_results.csv')
        df = pd.read_csv(filepath)
        df.columns = ['Description', 'Color Difference']  # 手动设置列名
        df['Digit'] = df['Description'].str.extract(r'(\d+)').astype(int)  # 从描述中提取数字
        df['Gen'] = subdir  # 添加生成代列
        dfs.append(df[['Digit', 'Color Difference', 'Gen']])
    combined_df = pd.concat(dfs)
    return combined_df

# 读取并合并数据
combined_df = read_color_diff_files(base_path, subdirectories)

# 数据透视
pivot_df = combined_df.pivot(index='Digit', columns='Gen', values='Color Difference')

# 绘图
plt.figure(figsize=(15, 6))
pivot_df.plot(kind='bar')
plt.xlabel('Digit')
plt.ylabel('Color Difference Value')
plt.title('Color Difference Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

# 保存图表
output_path = 'digitcolor_difference_values_bar_plot.png'
plt.savefig(output_path)
plt.close()

output_path
