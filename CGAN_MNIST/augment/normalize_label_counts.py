import pandas as pd
import numpy as np
import os
from random import randint
 
csv_path = "MNIST/train.csv"

df = pd.read_csv(csv_path)

# 设置目标图片数量
target_count = 6000
max_color_count = 2000

# 计算每个标签和颜色组合的图片数量
label_color_counts = df.groupby(['label', 'color']).size().reset_index(name='count')

# 调整每个标签和颜色的图片数量
adjusted_df = pd.DataFrame()
for index, row in label_color_counts.iterrows():
    label = row['label']
    color = row['color']
    subset = df[(df['label'] == label) & (df['color'] == color)]
    
    if row['count'] > max_color_count:
        # 如果某一颜色的图片数量超过2000，随机删除多余的图片
        subset = subset.sample(n=max_color_count, random_state=42)
    
    adjusted_df = pd.concat([adjusted_df, subset], ignore_index=True)

# 确保每个标签的总数不超过6000
final_df = pd.DataFrame()
for label in adjusted_df['label'].unique():
    label_subset = adjusted_df[adjusted_df['label'] == label]
    if len(label_subset) > target_count:
        label_subset = label_subset.sample(n=target_count, random_state=42)
    final_df = pd.concat([final_df, label_subset], ignore_index=True)

new_csv_path = os.path.join('augment', 'adjusted_train_dataset.csv')
final_df.to_csv(new_csv_path, index=False)

print(f"Adjusted dataset saved. Total records: {len(final_df)}.")

label_color_counts = final_df.groupby(['label', 'color']).size().reset_index(name='count')
  
print("label_color_counts \n",label_color_counts)