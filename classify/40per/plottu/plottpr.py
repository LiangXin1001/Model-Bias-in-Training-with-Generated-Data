import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base path
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'
subdirectories = ['gen0', 'gen1', 'gen2', 'gen3', 'gen4']

# Function to read TPR and FPR files
def read_tpr_fpr_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'tpr_fpr_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_tpr_fpr_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_tpr_df = combined_df.pivot_table(index='Digit', columns='Gen', values='TPR', aggfunc='mean')
pivot_fpr_df = combined_df.pivot_table(index='Digit', columns='Gen', values='FPR', aggfunc='mean')

# Plotting TPR
pivot_tpr_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0.4, 1)  # Adjust the y-axis limit if necessary
plt.xlabel('Digit')
plt.ylabel('TPR Value')
plt.title('TPR Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()
tpr_output_path = 'tpr_values_bar_plot.png'
plt.savefig(tpr_output_path)
plt.close()

# Plotting FPR
pivot_fpr_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0, 1)  # Adjust the y-axis limit if necessary
plt.xlabel('Digit')
plt.ylabel('FPR Value')
plt.title('FPR Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()
fpr_output_path = 'fpr_values_bar_plot.png'
plt.savefig(fpr_output_path)
plt.close()