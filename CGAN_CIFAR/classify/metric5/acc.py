import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import seaborn as sns  
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import sys
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D 
import matplotlib.pyplot as plt
current_directory = os.path.dirname(__file__)  
parent_directory = os.path.dirname(current_directory)  
sys.path.append(parent_directory)
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf ,CIFAR_100_CLASS_MAP,generate_full_subclass_map
import argparse
 
parser = argparse.ArgumentParser(description='Calculate Disparate Impact (DI) for each Superclass and Subclass combination')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--model_name', type=str, required=True, help='Model name used for saving results')
parser.add_argument('--output_path', type=str, required=True, help='Model name used for saving results')

args = parser.parse_args()
 

# 定义CSV文件路径
base_dir = args.result_dir

csv_files = [f'test_results_{i}.csv' for i in range(11)]  # 从0到10的文件名列表

# 初始化用于存储平均值和标准差的数组
num_positions = 5  # 最差到第五差
num_files = len(csv_files)
worst_accuracies_mean = np.zeros((num_positions, num_files))
worst_accuracies_std = np.zeros((num_positions, num_files))
 
for file_index, csv_file in enumerate(csv_files):
    csv_file_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_file_path, header=None)
       
    column_names = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID', 'True Superclass Name', 'True Subclass Name']
    df.columns = column_names
    header_rows = df['Image'] == 'Image'

    # 删除包含表头的行
    df = df[~header_rows]

    # 重置索引
    df = df.reset_index(drop=True)

    numeric_columns = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID']

    # 将这些列转换为数值类型，无法转换的值会变为 NaN
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
 
    # 尝试将数值列转换为数值类型，检测无法转换的值
    for col in numeric_columns:
        # 尝试将列转换为数值类型，无法转换的值会变为 NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 检查是否存在 NaN 值
        if df[col].isnull().any():
            # 找出无法转换的值所在的行
            invalid_rows = df[df[col].isnull()]
            # 获取这些行的索引和对应的值
            invalid_indices = invalid_rows.index.tolist()
            invalid_values = invalid_rows[col].tolist()
            # 抛出错误并提供详细信息
            raise ValueError(f"Column '{col}' has invalid values at rows {invalid_indices}: {invalid_values}")


    # 将数值列转换为整数类型
    df[numeric_columns] = df[numeric_columns].astype(int)

    # 验证数据
    print("Unique Run IDs:", df['Run ID'].unique())
    print(df.dtypes)

  

    # 获取所有的Run IDs
    run_ids = df['Run ID'].unique()
 
    num_runs = len(run_ids)
    
    # 初始化用于存储每次运行的准确率的列表
    worst_accuracies_runs = [[] for _ in range(num_positions)]  # 列表的列表

    # 分别处理每次运行
    for run_id in run_ids:
        df_run = df[df['Run ID'] == run_id]
        
        # 计算整体准确率
        overall_accuracy = accuracy_score(df_run['True Superclass'], df_run['Predicted Superclass'])
        print(f"{csv_file} - Run {run_id} - Overall Accuracy: {overall_accuracy:.2%}")

        #  Compute accuracies for each subclass
        
        for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
            subclass_accuracies = []
            print(f"Accuracy for subclasses in superclass {superclass}:")
            for subclass in subclasses:
              
                subclass_data = df_run[df_run['True Subclass Name'] == subclass]
                # print("subclass_data")
                # print(subclass_data)
                if len(subclass_data) == 0:
                    raise ValueError(f"No data found in model {args.model_name} for subclass '{subclass}' in file '{csv_file}', run ID '{run_id}'.")
                else:
                    accuracy = accuracy_score(subclass_data['True Superclass'], subclass_data['Predicted Superclass'])
                subclass_accuracies.append(accuracy)
                print(f"{subclass}: {accuracy:.2%}")

                # Sort subclass accuracies and collect worst to best
           
            sorted_accuracies = sorted(subclass_accuracies)
            # print("sorted_accuracies : ",sorted_accuracies)
            for pos in range(num_positions):
                worst_accuracies_runs[pos].append(sorted_accuracies[pos])

  
    for pos in range(num_positions):
        accuracies = worst_accuracies_runs[pos]
        mean_accuracy = np.nanmean(accuracies)
        std_accuracy = np.nanstd(accuracies)
        worst_accuracies_mean[pos, file_index] = mean_accuracy
        worst_accuracies_std[pos, file_index] = std_accuracy

    print(f"Mean and standard deviation of accuracies for {csv_file}:")
    for pos in range(num_positions):
        print(f"{pos+1} Worst Mean Accuracy: {worst_accuracies_mean[pos, file_index]:.2%}, Standard Deviation: {worst_accuracies_std[pos, file_index]:.2%}")

csv_file_indices = np.arange(len(csv_files))
# labels = ['The 1-th Worst', 'The 2-th Worst', 'The 3-th Worst', 'The 4-th Worst', 'The 5-th Worst'] 
labels = ['Worst', '2nd Worst', '3rd Worst', '4th Worst', '5th Worst']
colors = ['blue', 'green', 'red', 'purple', 'orange']


markers = ['o', 's', 'D', '^', 'v']  # 圆形, 方形, 菱形, 上三角, 下三角


# 假设 worst_accuracies_mean 和 worst_accuracies_std 已经被定义
# 构建数据为 DataFrame
data = []
csv_file_indices = np.arange(len(csv_files))

for i in range(num_positions):
    for j in range(len(csv_file_indices)):
        data.append({
            'Generation': j,
            'Accuracy': worst_accuracies_mean[i, j],
            'StdDev': worst_accuracies_std[i, j],
            'Position': labels[i]
        })

df = pd.DataFrame(data)

# 绘制带标准差阴影的曲线图
plt.figure(figsize=(12, 8))

# 遍历不同的位置来绘制每条曲线和标准差阴影
for i, label in enumerate(labels):
    position_data = df[df['Position'] == label]
    x = position_data['Generation']
    y = position_data['Accuracy']
    yerr = position_data['StdDev']
        
    # 拟合自回归线
    slope, intercept = np.polyfit(x, y, 1)  # 一次线性回归
    fit_line = slope * x + intercept
    
    # 绘制拟合曲线
    plt.plot(x, fit_line, label=f'{label} Regression', color=colors[i], linestyle='-', linewidth=2.5)
    plt.scatter(x, y, color=colors[i], label=f'{label} Data', s=30, alpha=0.7, marker=markers[i])
    # 绘制曲线
    # sns.lineplot(x=x, y=y, label=label, color=colors[i], linewidth=3)

    # 绘制标准差阴影
    plt.fill_between(x, y - yerr, y + yerr, color=colors[i], alpha=0.1)
 
# 设置图表属性
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title(f'{args.model_name}') 
plt.legend(title='Position')
plt.grid(True)
plt.tight_layout()

# 保存图像
os.makedirs(f'{args.output_path}/{args.model_name}', exist_ok=True)
output_plot_path = f'{args.output_path}/{args.model_name}/accuracy_trends_with_std.png'
plt.savefig(output_plot_path)
plt.show()

print(f"Image saved in {output_plot_path}")








# # Plotting accuracy trends with error bars and linear regression lines
# plt.figure(figsize=(12, 8))
# for i in range(num_positions):
#     # 计算线性回归参数
#     slope, intercept = np.polyfit(csv_file_indices, worst_accuracies_mean[i], 1)
#     # 生成拟合直线
#     fit_line = slope * csv_file_indices + intercept

#     # 绘制数据点和误差棒
#     plt.errorbar(csv_file_indices, worst_accuracies_mean[i], yerr=worst_accuracies_std[i],
#                  marker='o', color=colors[i], label=f'{labels[i]} accuracy', capsize=5, linestyle='none')
#     # 绘制线性回归拟合线
#     plt.plot(csv_file_indices, fit_line, color=colors[i], linestyle='-',
#              label=f'{labels[i]} Fit (y={slope:.4f}x + {intercept:.4f})', linewidth=3)

# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.title('Trend of Worst to Fifth Worst Accuracies Across CSV Files')
# plt.xticks(csv_file_indices, [f'Gen {i}' for i in range(len(csv_files))], rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
 
# os.makedirs(f'{args.output_path}/{args.model_name}', exist_ok=True)
# output_plot_path = f'{args.output_path}/{args.model_name}/accuracy_trends_regression.png'
# plt.savefig(output_plot_path)
# plt.show()

# print(f"image saved in  {output_plot_path}")






base_dir = os.path.dirname(args.result_dir)
# 定义模型名称和绘图显示名称
 
 
num_positions = 5  # 最差到第五差
num_files = 11     # 每个模型对应11个CSV文件（从0到10）
labels = ['Worst', '2nd Worst', '3rd Worst', '4th Worst', '5th Worst']
colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', 'D', '^', 'v']  # 不同的标记符号

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']  # 用于绘图标签

# 遍历模型，设置模型子路径
all_models_worst_accuracies_mean = []
all_models_worst_accuracies_std = []

for model_idx, model_name in enumerate(model_names):
    csvpath = os.path.join(base_dir, model_name)  # 将模型名称添加到路径
    csv_files = [f'test_results_{i}.csv' for i in range(11)]  # 假设每个模型有 11 个文件
    
    # 初始化数组
    worst_accuracies_mean = np.zeros((5, len(csv_files)))
    worst_accuracies_std = np.zeros((5, len(csv_files)))
    
    # 遍历每个文件
    for file_index, csv_file in enumerate(csv_files):
        csv_file_path = os.path.join(csvpath, csv_file)
        df = pd.read_csv(csv_file_path, header=None)

        # 设置列名并删除重复表头行
        column_names = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID', 'True Superclass Name', 'True Subclass Name']
        df.columns = column_names
        df = df[df['Image'] != 'Image'].reset_index(drop=True)  # 去除含表头内容的行

        # 将数值列转换为整数类型
        numeric_columns = ['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'Run ID']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').astype(int)

        # 获取 Run IDs
        run_ids = df['Run ID'].unique()
        worst_accuracies_runs = [[] for _ in range(5)]

        # 处理每个运行 ID
        for run_id in run_ids:
            df_run = df[df['Run ID'] == run_id]
            
            # 计算每个超类的子类准确率
            for superclass, subclasses in CIFAR_100_CLASS_MAP.items():
                subclass_accuracies = [accuracy_score(df_run[df_run['True Subclass Name'] == subclass]['True Superclass'],
                                                      df_run[df_run['True Subclass Name'] == subclass]['Predicted Superclass'])
                                       for subclass in subclasses if not df_run[df_run['True Subclass Name'] == subclass].empty]
                sorted_accuracies = sorted(subclass_accuracies)
                for pos in range(5):
                    worst_accuracies_runs[pos].append(sorted_accuracies[pos])

        # 计算均值和标准差
        for pos in range(5):
            accuracies = worst_accuracies_runs[pos]
            worst_accuracies_mean[pos, file_index] = np.nanmean(accuracies)
            worst_accuracies_std[pos, file_index] = np.nanstd(accuracies)

    # 保存每个模型的均值和标准差
    all_models_worst_accuracies_mean.append(worst_accuracies_mean)
    all_models_worst_accuracies_std.append(worst_accuracies_std)

# 现在使用 all_models_worst_accuracies_mean 和 all_models_worst_accuracies_std 进行绘图
fig, axes = plt.subplots(1, len(model_names), figsize=(20, 5), sharey=True)
fig.suptitle("Trend of Worst to Fifth Worst Accuracies Across Generations")

# 遍历每个模型
for model_idx, ax in enumerate(axes):
    model_name = model_names[model_idx]
    display_name = display_names[model_idx]
    worst_accuracies_mean = all_models_worst_accuracies_mean[model_idx]
    worst_accuracies_std = all_models_worst_accuracies_std[model_idx]

    # 绘制每个位置的曲线和标准差阴影
    for i, label in enumerate(labels):
        x = csv_file_indices
        y = worst_accuracies_mean[i]
        yerr = worst_accuracies_std[i]
        
        # 绘制原始数据点
        ax.scatter(x, y, color=colors[i], s=30, alpha=0.7, marker=markers[i], label=label)

        # 绘制拟合的回归线
        slope, intercept = np.polyfit(x, y, 1)
        fit_line = slope * x + intercept
        ax.plot(x, fit_line, color=colors[i], linestyle='--', linewidth=2)
        
        # 绘制标准差阴影
        ax.fill_between(x, y - yerr, y + yerr, color=colors[i], alpha=0.1)
        
        # 设置x轴网格线隔一个单位画一条
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔1个x轴值画一个网格线

        # 设置网格线样式
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

    
    # 设置子图标题和标签
    ax.set_title(display_name)
    ax.set_xlabel('Generation')
    ax.set_xticks(csv_file_indices)

axes[0].set_ylabel('Accuracy')
# 创建自定义的图例句柄，以确保图例与标记符号和颜色一致
legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=labels[i],
                          markerfacecolor=colors[i], markersize=8, linestyle='--')
                   for i in range(num_positions)]

# 在图的顶部添加自定义的全局图例
fig.legend(handles=legend_elements, loc='upper center', ncol=num_positions, bbox_to_anchor=(0.5, 1.15), frameon=False)

 
# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 调整图与标题和图例的距离

# 保存图像
output_plot_path = os.path.join(args.output_path, 'accuracy_trends_across_models.png')
plt.savefig(output_plot_path, bbox_inches='tight')
plt.show()

print(f"Image saved in {output_plot_path}")