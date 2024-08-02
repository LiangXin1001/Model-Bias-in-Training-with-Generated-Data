import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base path
base_path = '/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/classifier/results'

# Define subdirectories
subdirectories = ['gen0', 'gen1', 'gen2','gen3','gen4']
 
def read_eo_files(base_path, subdirectories):
    dfs = []
    for subdir in subdirectories:
        filepath = os.path.join(base_path, subdir, 'eo_results.csv')
        df = pd.read_csv(filepath)
        df['Gen'] = subdir
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df

# Read and combine data
combined_df = read_eo_files(base_path, subdirectories)

# Pivot the DataFrame for plotting
pivot_df = combined_df.pivot(index='Digit', columns='Gen', values='EO')

# Plotting
pivot_df.plot(kind='bar', figsize=(15, 6))
plt.ylim(0.85, 1.0)
plt.xlabel('Digit')
plt.ylabel('EO Value')
plt.title('EO Values for Different Digits Across Generations')
plt.legend(title='Generation')
plt.tight_layout()

output_path = 'eo_values_bar_plot.png'
plt.savefig(output_path)
plt.close()
