import numpy as np
import os
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io
import faiss
from sklearn.preprocessing import normalize
import random

# 1. Load data
print("Loading embeddings...")
data = np.load("/data/databases/MajorTom5T/outputs/embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

print("Loading centroids...")
centroids_data = np.load("/data/databases/MajorTom5T/outputs/centroids_faiss.npz")
centroids = centroids_data["centroids"].astype(np.float32)

base_dir = "/data/databases/Core-S2L2A"
print("Loading metadata...")
manifest = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"), columns=["grid_cell", "parquet_url", "parquet_row"])

# 2. Assign images to clusters
print("Sampling 50,000 images to find threshold...")
np.random.seed(42)
sampled_indices = np.random.choice(len(embeddings), 50000, replace=False)

# Normalize sampled embeddings and query FAISS for closest cluster
X_sampled = normalize(embeddings[sampled_indices], norm="l2")
index_cpu = faiss.IndexFlatL2(centroids.shape[1])
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
except:
    index = index_cpu
index.add(centroids)

print("Searching FAISS for cluster assignments...")
distances, labels = index.search(X_sampled, 1)
labels = labels.flatten()

# Find the pure black vector index to define garbage clusters (we found it was 1231300 but let's just re-find or hardcode)
# Black vector was at index 1231300
black_vector = embeddings[1231300]
black_vector_norm = normalize(black_vector.reshape(1, -1), norm="l2")
# Find similarity of all centroids to black vector
similarities = np.dot(centroids, black_vector_norm.T).flatten()

# Let's say a garbage cluster is one with >0.90 similarity to pure black
garbage_cluster_ids = set(np.where(similarities > 0.90)[0])
print(f"Identified {len(garbage_cluster_ids)} garbage (black) clusters.")

def bytes_to_array(b):
    if b is None: return np.zeros((256, 256))
    if isinstance(b, dict): b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

results = []

print("Extracting black pixel percentages...")
# Limit to processing 2000 images to save time, but ensuring we sample both garbage and non-garbage
garbage_indices = [i for i, lbl in enumerate(labels) if lbl in garbage_cluster_ids]
good_indices = [i for i, lbl in enumerate(labels) if lbl not in garbage_cluster_ids]

# Let's sample 500 garbage and 1000 good
to_process = random.sample(garbage_indices, min(500, len(garbage_indices))) + \
             random.sample(good_indices, min(1000, len(good_indices)))

for i, local_idx in enumerate(to_process):
    if i % 100 == 0: print(f"Processing {i}/{len(to_process)}")
    global_idx = sampled_indices[local_idx]
    cluster_id = labels[local_idx]
    is_garbage = cluster_id in garbage_cluster_ids
    
    row = manifest.iloc[global_idx]
    parquet_file = os.path.join(base_dir, "images", row['parquet_url'].split("/")[-1])
    row_group = int(row['parquet_row'])
    
    try:
        dataset = pq.ParquetFile(parquet_file)
        table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
        R = bytes_to_array(table.column('B04')[0].as_py())
        G = bytes_to_array(table.column('B03')[0].as_py())
        B = bytes_to_array(table.column('B02')[0].as_py())
        
        rgb = np.stack([R, G, B], axis=-1)
        zeros_pct = (rgb == 0).mean() * 100
        
        results.append({
            "idx": global_idx,
            "cluster": cluster_id,
            "is_garbage": is_garbage,
            "black_pct": zeros_pct
        })
    except Exception:
        continue

df = pd.DataFrame(results)

print("\n--- Distribution of Black Pixel % ---\n")
print("For Garbage Clusters:")
print(df[df.is_garbage]['black_pct'].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

print("\nFor Meaningful Clusters:")
print(df[~df.is_garbage]['black_pct'].describe(percentiles=[0.5, 0.75, 0.8, 0.9, 0.95, 0.99]))

df.to_csv("black_threshold_analysis.csv", index=False)
