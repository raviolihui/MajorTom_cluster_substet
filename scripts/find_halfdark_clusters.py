import numpy as np
import faiss
from sklearn.preprocessing import normalize
import os
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import io

print("Loading embeddings...")
data = np.load("/data/databases/MajorTom5T/outputs/embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

print("Loading centroids...")
centroids_data = np.load("/data/databases/MajorTom5T/outputs/centroids_faiss.npz")
centroids = centroids_data["centroids"].astype(np.float32)

print("Setting up FAISS index...")
index_cpu = faiss.IndexFlatL2(centroids.shape[1])
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    print("Using GPU FAISS")
except:
    index = index_cpu
    print("Using CPU FAISS")
index.add(centroids)

print("Loading metadata...")
base_dir = "/data/databases/Core-S2L2A"
manifest = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"), columns=["grid_cell", "parquet_url", "parquet_row"])

def bytes_to_array(b):
    if b is None: return np.zeros((256, 256))
    if isinstance(b, dict): b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

print("Scanning for half-dark images (~30% to ~70% black pixels)...")
np.random.seed(42)
test_indices = np.random.choice(len(manifest), 20000, replace=False)

half_dark_clusters = set()
found_count = 0

for idx in test_indices:
    row = manifest.iloc[idx]
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
        
        if 30 < zeros_pct < 70:
            print(f"\nFound image with {zeros_pct:.1f}% missing data at index {idx}!")
            
            emb = embeddings[idx].reshape(1, -1)
            emb_norm = normalize(emb, norm="l2")
            D, I = index.search(emb_norm, 1)
            cluster_id = I[0][0]
            
            print(f" -> Assigned to cluster: {cluster_id} (Distance: {D[0][0]:.4f})")
            half_dark_clusters.add(cluster_id)
            found_count += 1
            
            if found_count >= 10:
                break
    except Exception as e:
        continue

print("\n--- Summary ---")
print("Target Clusters to Inspect:", list(half_dark_clusters))
