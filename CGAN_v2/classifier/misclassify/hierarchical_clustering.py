import clip
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import argparse
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Process misclassified images using CLIP and hierarchical clustering.')
parser.add_argument('--base_dir', type=str, required=True, help='Root directory where images are stored.')
args = parser.parse_args()

root_dir = os.path.join(args.base_dir, 'misclassified_images')
csv_file = os.path.join(args.base_dir, 'misclassified_images.csv')
output_dir = os.path.join(args.base_dir, 'clustering_results')
os.makedirs(output_dir, exist_ok=True) 

class MisclassifiedDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or Compose([
            Resize((224, 224)),  # CLIP的输入大小
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
dataset = MisclassifiedDataset(csv_file= csv_file, root_dir= root_dir, transform=preprocess)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
def get_clip_features(model, data_loader, device):
    features = []
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            feature = model.encode_image(images)
            features.append(feature)
    return torch.cat(features).cpu().numpy()
 
features = get_clip_features(model, loader, device)



# use Hierarchical Clustering
def print_clusters(data_frame, linkage_matrix, max_distance):
    labels = fcluster(linkage_matrix, max_distance, criterion='distance')
    unique_clusters = np.unique(labels)
    print(f"Clusters at distance {max_distance}:")
    for cluster in unique_clusters:
        members = data_frame['Filename'][labels == cluster].tolist()
        print(f"Cluster {cluster}: {members}")

data_frame = pd.read_csv(csv_file)          
linked = linkage(features, 'ward')

print_clusters(data_frame, linked, 20)
print_clusters(data_frame, linked, 15)

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=list(range(len(features))),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig(os.path.join( output_dir, 'dendrogram.png'), format='png', dpi=300)
plt.show()
