import pandas as pd
 
from collections import Counter
from PIL import Image, ImageFilter
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--base_image_dir', type=str, required=True)
parser.add_argument('--csv_filename', type=str, required=True)

args = parser.parse_args()

 

# Load the CSV file
csv_path = os.path.join(args.base_image_dir, args.csv_filename)
df = pd.read_csv(csv_path)
 
 


# Function to detect the most common color
def detect_color(image_name):
   # Extract class from image name
    class_folder = image_name.split('_')[0]+'_'+image_name.split('_')[1]
    image_path = os.path.join(args.base_image_dir, class_folder, image_name)


    with Image.open(image_path) as img:
        # Apply a median filter to reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.convert('RGB')
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        # Assuming the background (likely black) should not be counted
        color_counts.pop((0, 0, 0), None)
        # Find the most common color
        if color_counts:
            common_color = max(color_counts, key=color_counts.get)
            if common_color[0] > common_color[1] and common_color[0] > common_color[2]:
                return 0  # Red
            elif common_color[1] > common_color[0] and common_color[1] > common_color[2]:
                return 2  # Green
            else:
                return 1  # Blue
        return None

# Apply color detection to each image and store the results
df['color'] = df['image_name'].apply(detect_color)
 
# Save the updated CSV
df.to_csv(csv_path, index=False)

print("CSV file has been updated with color labels.")
