import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# Define the base path
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'
subdirectories = ['gen{}'.format(i) for i in range(11)]

# Function to read digit difference files
def read_digit_diff_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'digit_difference_results.csv')
        df = pd.read_csv(filepath, header=None, names=['Description', 'Value'])
        df['Gen'] = subdir  # Add generation column
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_digit_diff_files(base_path, subdirectories)


# Extract the color and digit difference from the Description
combined_df = combined_df.dropna(subset=['Description'])
# print("combined_df")
# print(combined_df)
combined_df['Color Digit Difference'] = combined_df['Description'].apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])

# Pivot the DataFrame for calculating means
pivot_df = combined_df.pivot_table(index='Gen', columns='Color Digit Difference', values='Value', aggfunc='mean')

# Ensure the Gen column is numeric and sort by it
pivot_df.index = pivot_df.index.str.extract('(\d+)').astype(int).squeeze()
pivot_df = pivot_df.sort_index()

# Plotting the average digit differences across generations
plt.figure(figsize=(10, 6))
for col in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[col], marker='o', label=col)
plt.ylim(0,0.2)
plt.xlabel('Generation')
plt.ylabel('Average Digit Difference')
plt.title('Average Digit Difference Across Generations')
plt.legend(title='Color Digit Difference')
plt.grid(True)

# Save the plot
output_path_curve = 'average_digit_difference_curve.png'
plt.savefig(output_path_curve)
plt.close()

print(f'Average digit difference curve plot saved as {output_path_curve}')
