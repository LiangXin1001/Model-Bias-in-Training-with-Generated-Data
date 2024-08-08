# import pandas as pd

# # 指定CSV文件的路径
# csv_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/augment/augmented_train.csv'  # 修改为你的CSV文件路径

# # 从CSV文件加载数据
# df = pd.read_csv(csv_path)

# # 检查 color 列是否有缺失值，并从 image_name 中提取 color 填充缺失值
# if df['color'].isnull().any():
#     df['color'] = df.apply(
#         lambda row: int(row['image_name'].split('_')[2]) if pd.isnull(row['color']) else row['color'], axis=1)

# # 显示更新后的DataFrame，以验证填充结果
# print(df)

# # 可以选择保存修正后的DataFrame到新的CSV文件
# df.to_csv('/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/augment/train.csv', index=False)  # 修改为你希望保存的路径


import pandas as pd

# 指定CSV文件的路径
csv_path = './augment/train.csv'

# 从CSV文件加载数据
df = pd.read_csv(csv_path)

# 检查和确保每个 color 是 int 类型
def ensure_int_color(row):
    try:
        # 尝试将color转换为int
        int_color = int(row['color'])
    except ValueError:
        # 如果转换失败，从image_name中提取color
        int_color = int(row['image_name'].split('_')[2])
    return int_color

df['color'] = df.apply(ensure_int_color, axis=1)

# 显示更新后的DataFrame，以验证填充结果
print(df)

# 可以选择保存修正后的DataFrame到新的CSV文件
df.to_csv('./augment/train1.csv', index=False)
