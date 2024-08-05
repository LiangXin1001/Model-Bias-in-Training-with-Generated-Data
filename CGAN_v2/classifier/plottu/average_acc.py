import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# Define the base path
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read all images result files
def read_all_images_result_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'all_images_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir  # Add generation column
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_all_images_result_files(base_path, subdirectories)

# Calculate accuracy for each row
combined_df['Correct'] = combined_df['True Label'] == combined_df['Predicted Label']

# Calculate the average accuracy for each gen
average_accuracy_per_gen = combined_df.groupby('Gen')['Correct'].mean()

# Ensure the Gen column is numeric and sort by it
average_accuracy_per_gen.index = average_accuracy_per_gen.index.str.extract('(\d+)').astype(int).squeeze()
average_accuracy_per_gen = average_accuracy_per_gen.sort_index()

# Plot the average accuracy values across generations
plt.figure(figsize=(10, 6))
plt.plot(average_accuracy_per_gen.index, average_accuracy_per_gen.values, marker='o')
plt.ylim(0.8, 1.0)
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations')
plt.xticks(average_accuracy_per_gen.index)  # Ensure x-axis shows all gen values
plt.grid(True)

# Save the plot
output_path_curve = 'average_accuracy_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average accuracy curve plot saved as {output_path_curve}')



# 绘制散点图并添加线性回归直线
x = average_accuracy_per_gen.index
y = average_accuracy_per_gen.values

# 计算线性回归
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Average Accuracy')
plt.plot(x, regression_line, color='red', label=f'Linear Regression\n(y={slope:.4f}x + {intercept:.4f})')
plt.ylim(0.8, 1.0)
plt.xlabel('Generation')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy Across Generations')
plt.xticks(x)  # 确保 x 轴显示所有代值
plt.grid(True)
plt.legend()

# 保存图像
output_path_scatter = 'average_accuracy_scatter_linear.png'
plt.savefig(output_path_scatter)
plt.close()

print(f'Average accuracy scatter plot with linear regression saved as {output_path_scatter}')