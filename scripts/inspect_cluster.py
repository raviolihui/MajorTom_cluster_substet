import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
from sklearn.preprocessing import normalize
from PIL import Image
import io
import matplotlib.pyplot as plt

# Target specific cluster
TARGET_CLUSTERS = [1822, 3138, 4137, 85976] #[1822, 3138, 4137, 85976] #[132, 1822, 3138, 6576, 11328, 20386, 34883, 39753, 44900, 46970]

# Configuration
embeddings_path = "/data/databases/MajorTom5T/outputs_filtered/embeddings_filtered.npz"
centroids_path = "/data/databases/MajorTom5T/outputs_filtered/centroids_faiss.npz"
meta_path = "/data/databases/MajorTom5T/outputs_filtered/full_dataset_manifest.parquet"
base_dir = "/data/databases/Core-S2L2A"

print(f"Loading embeddings and metadata to find clusters {TARGET_CLUSTERS}...")
data = np.load(embeddings_path, allow_pickle=True)
embeddings = data["embeddings"].astype(np.float32)

manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])
def _to_local_path(url: str) -> str:
    return os.path.join(base_dir, "images", url.split("/")[-1])
manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
manifest["row_group"] = manifest["parquet_row"].astype(int)

if "indices" in data:
    print("Filtering manifest using valid indices from embeddings...")
    valid_indices = data["indices"]
    manifest = manifest.iloc[valid_indices].reset_index(drop=True)

print("Normalizing embeddings and assigning clusters using FAISS...")
X = normalize(embeddings, norm="l2")
centroids_data = np.load(centroids_path)
centroids = centroids_data["centroids"].astype(np.float32)

index_cpu = faiss.IndexFlatL2(X.shape[1])
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    print("FAISS is using GPU for search.")
except Exception as e:
    print(f"FAISS GPU not available, using CPU ({e})")
    index = index_cpu

index.add(centroids)
distances, labels = index.search(X, 1)
distances = distances.flatten()
labels = labels.flatten()

def bytes_to_array(b):
    if isinstance(b, dict):
        b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

for target_cluster in TARGET_CLUSTERS:
    output_dir = f"/home/carmenoliver/my_projects/Make_MajorTom_sbst/cluster_img/cluster_{target_cluster}_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images for our target cluster
    pts = np.where(labels == target_cluster)[0]
    if len(pts) == 0:
        print(f"\nNo images found in Cluster {target_cluster}.")
        continue
        
    cluster_dists = distances[pts]

    # Sort by distance (closest to centroid first)
    closest_relative_indices = np.argsort(cluster_dists)
    closest_absolute_indices = pts[closest_relative_indices]

    print(f"\nFound {len(pts)} images in Cluster {target_cluster}.")
    print(f"Extracting images to {output_dir}...\n")

    MAX_IMAGES = 20
    plot_count = min(len(pts), MAX_IMAGES)

    # Plotting setup for a grid limit to MAX_IMAGES
    cols = 6
    rows = (plot_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    
    # Handle case with only 1 row/col cleanly
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
        
    fig.suptitle(f"Top {plot_count} of {len(pts)} Images in Cluster {target_cluster}\n(Top-Left is the Centroid Sample)", fontsize=16)

    for i, idx in enumerate(closest_absolute_indices[:MAX_IMAGES]):
        dist = cluster_dists[closest_relative_indices[i]]
        row = manifest.iloc[idx]
        
        # Load image
        dataset = pq.ParquetFile(row['parquet_file'])
        table = dataset.read_row_group(row['row_group'], columns=['B04', 'B03', 'B02'])
        
        R = bytes_to_array(table.column('B04')[0].as_py())
        G = bytes_to_array(table.column('B03')[0].as_py())
        B = bytes_to_array(table.column('B02')[0].as_py())
        
        rgb = np.dstack((R, G, B)) / 3000.0
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        # The first image (i=0) is the selected centroid
        title = f"Dist: {dist:.4f}"
        if i == 0:
            title = f"** SELECTED CENTROID **\nDist: {dist:.4f}"
            
        axes[i].imshow(rgb)
        axes[i].set_title(title, fontsize=10 if i>0 else 12, color='red' if i==0 else 'black')
        axes[i].axis('off')

    # Hide any empty subplots
    for j in range(plot_count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out_file = os.path.join(output_dir, f"cluster_{target_cluster}_grid.png")
    plt.savefig(out_file, dpi=150)
    plt.close(fig) # IMPORTANT: Prevents matplotlib from accumulating figures and consuming all RAM
    print(f"✅ Saved grid visualization to: {out_file}")
