import os
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--train_csv_filename', type=str, required=True)
parser.add_argument('--generated_csv_path', type=str, required=True)
parser.add_argument('--new_csv_name', type=str, required=True)

args = parser.parse_args()

# 定义文件路径
base_image_dir = '/local/scratch/hcui25/Project/xin/CS/GAN'
train_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch/data',args.train_csv_filename)
generated_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch',args.generated_csv_path)

# 读取 CSV 文件
train_df = pd.read_csv(train_csv_path)
generated_df = pd.read_csv(generated_csv_path)

label_counts = train_df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']

# 计算每个数字需要增加的图片数量（50%）
label_counts['additional'] = np.ceil(label_counts['count'] * 0.40).astype(int)
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

# 检查和调整数据集大小
max_images = 180000
if len(combined_df) > max_images:
    # 需要删除的图片数量
    num_to_remove = len(combined_df) - max_images
    # 按优先级删除图片
    for gen_suffix in ['gen1.png', 'gen2.png', 'gen3.png','gen4.png','gen5.png','gen6.png','gen7.png','gen8.png','gen9.png','gen10.png']:
        if num_to_remove <= 0:
            break
        # 找到匹配 gen_suffix 结尾的图片
        gen_images = combined_df[combined_df['image_name'].str.endswith(gen_suffix)]
        if not gen_images.empty:
            # 计算删除的数量
            num_to_remove_gen = min(len(gen_images), num_to_remove)
            # 删除图片
            combined_df = combined_df.drop(gen_images.index[:num_to_remove_gen])
            num_to_remove -= num_to_remove_gen

# 保存新的 CSV 文件
new_csv_path = os.path.join(base_image_dir, 'CGAN-PyTorch/data',  args.new_csv_name)
combined_df.to_csv(new_csv_path, index=False)

print("新的 CSV 文件已生成并保存在:", new_csv_path)
