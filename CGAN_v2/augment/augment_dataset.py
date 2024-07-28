import os
import pandas as pd
import numpy as np

# 定义文件路径
base_image_dir = '/local/scratch/hcui25/Project/xin/CS/GAN'
train_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch/data','combined_train_gen0_gen1.csv')
generated_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch/generated_images_gen2', 'gen2_test.csv')

# 读取 CSV 文件
train_df = pd.read_csv(train_csv_path)
generated_df = pd.read_csv(generated_csv_path)

label_counts = train_df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']

# 计算每个数字需要增加的图片数量（20%）
label_counts['additional'] = np.ceil(label_counts['count'] * 0.20).astype(int)
print("label_counts \n",label_counts)
 

# 提取必要的图片从 generated_df
additional_images = pd.DataFrame(columns=['image_name', 'label', 'color'])

for _, row in label_counts.iterrows():
    label = row['label']
    
    required = row['additional']
    
    match_images = generated_df[(generated_df['label'] == label)]
 
    sampled_images = match_images.sample(n=min(required, len(match_images)), random_state=42)
 
    additional_images = pd.concat([additional_images, sampled_images], ignore_index=True)
        
 
# 合并新旧数据
combined_df = pd.concat([train_df, additional_images], ignore_index=True)

# 保存新的 CSV 文件
new_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch/data','combined_train_gen1_gen2.csv')
combined_df.to_csv(new_csv_path, index=False)

print("新的 CSV 文件已生成并保存在:", new_csv_path)
