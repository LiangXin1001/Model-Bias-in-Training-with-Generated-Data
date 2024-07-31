 
import os
import pandas as pd
import matplotlib.pyplot as plt

 
# Define the base path
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'
subdirectories = ['gen0', 'gen1', 'gen2', 'gen3', 'gen4']

# Function to read disparate impact files
def read_di_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'disparate_impact_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_di_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_df = combined_df.pivot_table(index='Digit', columns='Gen', values='Disparate Impact', aggfunc='mean')

# Plotting
pivot_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0, 2)  # Adjust the y-axis limit if necessary
plt.xlabel('Digit')
plt.ylim(0.85, 1.0)
plt.ylabel('Disparate Impact Value')
plt.title('Disparate Impact Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

output_path = 'disparate_impact_values_bar_plot.png'
plt.savefig(output_path)
plt.close()
