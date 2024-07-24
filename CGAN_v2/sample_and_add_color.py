import pandas as pd
import argparse

# 定义颜色映射
color_mapping = {0: 0, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 2, 7: 2, 8: 1, 9: 2}# 10个数字对应的颜色
#  2	2	0	0	2	0	2	2	0	1
def sample_and_add_color(input_csv1, input_csv2, output_csv, num_samples=500):
    # 读取输入CSV文件
    df_input1 = pd.read_csv(input_csv1)
    df_input2 = pd.read_csv(input_csv2)
    
    # 按标签进行分组
    grouped = df_input1.groupby('label')

    # 存储采样后的数据
    sampled_data = []

 
    for label, group in grouped:
        sampled_group = group.sample(n=num_samples, replace=True)  # 进行采样
        sampled_group['color'] =  color_mapping[label]
        sampled_data.append(sampled_group)


    # 将采样后的数据合并为一个DataFrame
    df_sampled = pd.concat(sampled_data)

    
    # 将采样后的数据添加到第二个输入CSV文件中
    df_output = pd.concat([df_input2, df_sampled], ignore_index=True)

    # 保存新CSV文件
    df_output.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample images and add color column.')
    parser.add_argument('--input_csv1', type=str, required=True, help='Path to the first input CSV file.')
    parser.add_argument('--input_csv2', type=str, required=True, help='Path to the second input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples per label.')

    args = parser.parse_args()

    sample_and_add_color(args.input_csv1, args.input_csv2, args.output_csv, args.num_samples)
