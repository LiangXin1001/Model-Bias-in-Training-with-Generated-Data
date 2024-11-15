import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import seaborn as sns  
  
from matplotlib.lines import Line2D 
# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process test results for different models.')
parser.add_argument('--result_dir', type=str, required=True, help='Directory containing result CSV files')
parser.add_argument('--output_path', type=str, required=True, help='Path to save resulting plots')
args = parser.parse_args()

# 定义模型名称和显示名称
model_names = ['simplecnn', 'alexnet', 'mobilenetv3', 'vgg19', 'resnet50']
display_names = ['LeNet', 'AlexNet', 'MobileNet-V3', 'VGG-19', 'ResNet-50']  # 用于绘图标签

# 创建保存图像的目录
output_dir = args.output_path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to read all images result files and extract generation from filenames
def read_all_images_result_files(base_path, subdirectories, model_name):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, model_name, f'gen{subdir}', 'all_images_results.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Gen'] = int(subdir)  # Add generation column
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined_df

def process_model_data(model_name):
    base_path = args.result_dir
    subdirectories = [f'{i}' for i in range(11)]
    
    # Read and combine data
    combined_df = read_all_images_result_files(base_path, subdirectories, model_name)

    if combined_df.empty:
        print(f"No data found for model {model_name}")
        return pd.DataFrame()  # 返回空的 DataFrame

    # Calculate accuracy for each row
    combined_df['Correct'] = combined_df['True Label'] == combined_df['Predicted Label']

    # Filter `Run ID` between 0 and 4, and calculate the average accuracy and standard deviation for each generation
    filtered_df = combined_df[combined_df['Run ID'].between(0, 4)]
    stats_per_gen = filtered_df.groupby('Gen')['Correct'].agg(['mean', 'std']).reset_index()
    return stats_per_gen

colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', 'D', '^', 'v']

# Plot the average accuracy values across generations with standard deviation shadow
plt.figure(figsize=(10, 6))
for model_idx, model_name in enumerate(model_names):
    stats_per_gen = process_model_data(model_name)
    if not stats_per_gen.empty:
        x = stats_per_gen['Gen']
        y = stats_per_gen['mean']
        yerr = stats_per_gen['std']

        # 绘制原始数据点
        plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                     elinewidth=2, capsize=4, label=display_names[model_idx])

        # 绘制标准差阴影
        plt.fill_between(x, y - yerr, y + yerr, color=colors[model_idx], alpha=0.1)

plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations for Multiple Models')
plt.xticks(range(11))  # Ensure x-axis shows all generation values
plt.legend()
plt.grid(True)

# Save the plot
output_path_curve = os.path.join(output_dir, 'average_accuracy_curve_with_std.png')
plt.savefig(output_path_curve)
plt.close()

print(f'Average accuracy curve plot with standard deviation shadow saved as {output_path_curve}')

# Plotting scatter and adding linear regression line
plt.figure(figsize=(10, 6))
color_index = 0

for model_idx, model_name in enumerate(model_names):
    stats_per_gen = process_model_data(model_name)
    if not stats_per_gen.empty:
        x = stats_per_gen['Gen']
        y = stats_per_gen['mean']
        yerr = stats_per_gen['std']
        # 绘制原始数据点
        plt.scatter(x, y, color=colors[model_idx], label=f'{display_names[model_idx]} Average Accuracy', marker=markers[model_idx])

        # 计算线性回归
        slope, intercept = np.polyfit(x, y, 1)
        regression_line = slope * x + intercept

        # 绘制拟合的回归线
        plt.plot(x, regression_line, color=colors[model_idx], linestyle='-', linewidth=3, label=f'{display_names[model_idx]} Regression\n(y={slope:.4f}x + {intercept:.4f})')
          
        # 绘制原始数据点
        plt.errorbar(x, y, yerr=yerr, fmt=markers[model_idx], color=colors[model_idx], ecolor='lightgray',
                    elinewidth=2, capsize=4, label=display_names[model_idx])

        # 绘制标准差阴影
        plt.fill_between(x, y - yerr, y + yerr, color=colors[model_idx], alpha=0.1)



plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations')
legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=f'{display_names[i]}',
                          markerfacecolor=colors[i], markersize=8, linestyle='-')
                   for i in range(len(model_names))]
plt.legend(handles=legend_elements, loc='upper center', ncol=len(model_names), bbox_to_anchor=(0.5, 1.34), frameon=False)

plt.xticks(range(11))  # Ensure x-axis shows all generation values
plt.grid(True)
 

# Save the scatter plot with linear regression
output_path_scatter = os.path.join(output_dir, 'average_accuracy_scatter_linear.png')
plt.savefig(output_path_scatter)
plt.close()

print(f'Average accuracy scatter plot with linear regression saved as {output_path_scatter}')