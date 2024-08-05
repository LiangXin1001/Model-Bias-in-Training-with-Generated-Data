import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Train a model with configurable datasets')
parser.add_argument('--misclassified_csv_path', type=str, required=True, help='CSV file path for testing dataset')
parser.add_argument('--images_directory', type=str, required=True, help='Directory path for testing images')
parser.add_argument('--output_csv_path', type=str, required=True, help='Directory path to save models')
 
args = parser.parse_args()

 
misclassified_df = pd.read_csv(args.misclassified_csv_path)
 
misclassified_info = {}
for index, row in misclassified_df.iterrows():
    misclassified_info[row['Filename']] = {'True Label': row['True Label'], 'Predicted Label': row['Predicted Label']}

 
all_images_results = []

 
for filename in os.listdir(args.images_directory):
    if filename.endswith(".png"):   
        true_label = int(filename.split('_')[1])   
        color = int(filename.split('_')[2])   
        if filename in misclassified_info:
            predicted_label = misclassified_info[filename]['Predicted Label']
        else:     
            predicted_label = true_label
     
        all_images_results.append({
            'Filename': filename,
            'True Label': true_label,
            'Predicted Label': predicted_label,
            'Color': color   
        })

 
results_df = pd.DataFrame(all_images_results)

 
results_df.to_csv(args.output_csv_path, index=False)
