import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import pickle as pkl




def calculate_worst_accuracy(base_path, model_name, generations, worst_n=5):
    data = []
    for gen in generations:
        result_path = f"{base_path}/{model_name}/gen{gen}/all_images_results.csv"
        df = pd.read_csv(result_path)
        df['Correct'] = df['True Label'] == df['Predicted Label']  # 是否预测正确

        # 按准确性排序，选择最差的N个准确率
        sorted_df = df.sort_values(by='Correct', ascending=True)
        worst_accuracies_all = sorted_df['Correct'].head(worst_n).mean()  # 取前worst_n个最差准确率
        
        # 计算线性拟合
        csv_file_indices = np.array([gen])  # 生成代数
        slope, intercept = np.polyfit(csv_file_indices, [worst_accuracies_all], 1)  # 拟合斜率和截距
        
        data.append((csv_file_indices, [worst_accuracies_all], slope, intercept))
    
    # 保存数据到pkl文件
    output_path = f'./data/plot_worst_{model_name}_data.pkl'
    with open(output_path, 'wb') as f:
        pkl.dump(data, f)
    print(f'Data saved for {model_name} at {output_path}')

base_path = 'results'
generations = range(11)  # 假设有11代

# 针对每个模型计算最差准确率并保存
# model_names = ['VGG-19', 'ResNet-50', 'AlexNet', 'SimpleNet', 'MobileNet-V3']
model_names = ['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'simplecnn']
for model_name in model_names:
    calculate_worst_accuracy(base_path, model_name, generations)

# vgg19 = pkl.load(open('./data/plot_worst_vgg19_data.pkl', 'rb'))
# resnet50 = pkl.load(open('./data/plot_worst_resnet50_data.pkl', 'rb'))
# alexnet = pkl.load(open('./data/plot_worst_alexnet_data.pkl', 'rb'))
# simplenet = pkl.load(open('./data/plot_worst_simplecnn_data.pkl', 'rb'))
# mobilenetv3 = pkl.load(open('./data/plot_worst_mobilenetv3_data.pkl', 'rb'))


fontsize=20

# one row four subplots
fig, axs = plt.subplots(1, 5, figsize=(30, 5))
axs[0].set_title('VGG-19', fontsize=fontsize)
axs[1].set_title('ResNet-50', fontsize=fontsize)
axs[2].set_title('AlexNet', fontsize=fontsize)
axs[3].set_title('SimpleNet', fontsize=fontsize)
axs[4].set_title('MobileNet-V3', fontsize=fontsize)

axs[0].set_xlabel('Generation', fontsize=fontsize)
axs[1].set_xlabel('Generation', fontsize=fontsize)
axs[2].set_xlabel('Generation', fontsize=fontsize)
axs[3].set_xlabel('Generation', fontsize=fontsize)
axs[4].set_xlabel('Generation', fontsize=fontsize)


# make the space between the subfigure and the top line larger
plt.subplots_adjust(top=0.8, bottom=0.2, left=0.05, right=0.95,)


# four lines, each line has a different color and marker and linestyle
markers = ['o','D','s','v','*', 'x', 'p', 'P', 'h', 'H', '+', 'X', 'D', 'd', '|', '_']
# use more clear different line styles
colors = ['blue', 'green', 'red', 'purple', 'orange', 'black', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime', 'magenta', 'navy', 'teal', 'yellow']


model_names = ['VGG-19','ResNet-50','AlexNet','SimpleNet','MobileNet-V3']


data = [vgg19, resnet50, alexnet, simplenet, mobilenetv3]

for i in range(5):
    for j in range(5):
        csv_file_indices, worst_accuracies_all, slope, intercept =  data[i][j]
        slope, intercept = np.polyfit(csv_file_indices, worst_accuracies_all, 1)
        # 生成拟合直线
        fit_line = slope * csv_file_indices + intercept

        # 绘制数据点和拟合直线
        axs[i].scatter(csv_file_indices, worst_accuracies_all, color=colors[j])
        axs[i].plot(csv_file_indices, fit_line, color=colors[j], label='The {}-th Worst'.format(j+1))
    
    axs[i].xaxis.set_tick_params(labelsize=fontsize)
    axs[i].yaxis.set_tick_params(labelsize=fontsize)
    
    axs[i].grid(True)

plt.legend(ncol=5,loc = 'upper center',bbox_to_anchor=(-1.3, 1.35),fontsize=fontsize)
plt.tight_layout()
plt.savefig('./imgs/nopretrain_cifar_worst_plot.png')
plt.savefig('./imgs/nopretrain_cifar_worst_plot.pdf')
