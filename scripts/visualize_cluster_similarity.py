import os
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
from sklearn.preprocessing import normalize
from PIL import Image
import io
import matplotlib.pyplot as plt

# Configuration
embeddings_path = "/data/databases/MajorTom5T/outputs/embeddings.npz"
centroids_path = "/data/databases/MajorTom5T/outputs/centroids_faiss.npz"
meta_path = "/data/databases/Core-S2L2A/metadata.parquet"
base_dir = "/data/databases/Core-S2L2A"
output_dir = "/home/carmenoliver/my_projects/Make_MajorTom_sbst/cluster_samples"
os.makedirs(output_dir, exist_ok=True)

print("Loading embeddings and full manifest...")
data = np.load(embeddings_path, allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])

def _to_local_path(url: str) -> str:
    fname = url.split("/")[-1]
    return os.path.join(base_dir, "images", fname)

manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
manifest["row_group"] = manifest["parquet_row"].astype(int)

# Normalize the embeddings like we did during clustering
print("Normalizing embeddings...")
X = normalize(embeddings, norm="l2")

print("Loading centroids...")
centroids_data = np.load(centroids_path)
centroids = centroids_data["centroids"].astype(np.float32)
d = X.shape[1]

# Use FAISS to recalculate the assignments for all 2.2M points
print("Assigning all points to their nearest centroids...")
index = faiss.IndexFlatL2(d)
index.add(centroids)

# Find the distance and cluster label for every point
distances, labels = index.search(X, 1)
distances = distances.flatten()
labels = labels.flatten()

# Pick 3 random clusters that have at least 5 images
num_clusters_to_view = 3
counts = np.bincount(labels, minlength=len(centroids))
valid_clusters = np.where(counts >= 5)[0]
random_clusters = random.sample(list(valid_clusters), num_clusters_to_view)

print(f"Selected random clusters: {random_clusters}")

def bytes_to_array(b):
    if isinstance(b, dict):
        b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

for cluster_id in random_clusters:
    # Get all indices for this cluster
    pts = np.where(labels == cluster_id)[0]
    cluster_dists = distances[pts]
    
    # Sort to get the 5 closest
    closest_relative_indices = np.argsort(cluster_dists)[:5]
    closest_absolute_indices = pts[closest_relative_indices]
    
    print(f"\nProcessing Cluster {cluster_id} with {len(pts)} total candidates")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Top 5 Closest Images for Cluster {cluster_id}", fontsize=16)
    
    for i, idx in enumerate(closest_absolute_indices):
        dist = cluster_dists[closest_relative_indices[i]]
        row = manifest.iloc[idx]
        pq_path = row['parquet_file']
        row_group = row['row_group']
        
        # Load image bands
        dataset = pq.ParquetFile(pq_path)
        table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
        
        R = bytes_to_array(table.column('B04')[0].as_py())
        G = bytes_to_array(table.column('B03')[0].as_py())
        B = bytes_to_array(table.column('B02')[0].as_py())
        
        rgb = np.dstack((R, G, B)) / 3000.0
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        axes[i].imshow(rgb)
        axes[i].set_title(f"Dist: {dist:.4f}\nRow: {idx}")
        axes[i].axis('off')
        
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}_top5.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

print(f"\nAll done! Check {output_dir} for the results.")
