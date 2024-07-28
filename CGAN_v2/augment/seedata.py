import os
import pandas as pd
import numpy as np
 
 

train_origen = "/local/scratch/hcui25/Project/xin/CS/GAN/CGAN-PyTorch/data/combined_train_gen1_gen2.csv"

train_df = pd.read_csv(train_origen)


label_counts = train_df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']
 
print("now train label_counts \n",label_counts)

label_color_counts = train_df.groupby(['label', 'color']).size().reset_index(name='count')
 
label_color_counts['additional'] = np.ceil(label_color_counts['count'] * 0.20).astype(int)
print("label_color_counts \n",label_color_counts)

