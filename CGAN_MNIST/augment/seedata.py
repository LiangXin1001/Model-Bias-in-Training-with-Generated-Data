import os
import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--csv_filename', type=str, required=True)
 
args = parser.parse_args()
 
  
train_origen = args.csv_filename
  
train_df = pd.read_csv(train_origen)


label_counts = train_df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']
 
print("now train label_counts \n",label_counts)

label_color_counts = train_df.groupby(['label', 'color']).size().reset_index(name='count')
 
label_color_counts['additional'] = np.ceil(label_color_counts['count'] * 0.20).astype(int)
print("label_color_counts \n",label_color_counts)

