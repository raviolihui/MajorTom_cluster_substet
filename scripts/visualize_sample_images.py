import os
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import io

output_dir = "/home/carmenoliver/my_projects/Make_MajorTom_sbst/sample_images"
os.makedirs(output_dir, exist_ok=True)

# Load your new FAISS subset manifest
manifest_path = "/data/databases/MajorTom5T/outputs/subset_manifest_faiss.parquet"
print(f"Loading manifest: {manifest_path}")
df = pd.read_parquet(manifest_path)

# Pick 5 random rows to visualize
num_samples = 5
sample_indices = random.sample(range(len(df)), num_samples)

print(f"Extracting {num_samples} random images from the dataset...")

for i, idx in enumerate(sample_indices):
    row = df.iloc[idx]
    pq_path = row['parquet_file']
    row_group = row['row_group']
    cluster_id = row['cluster_id']
    grid_cell = row['grid_cell']
    
    print(f"[{i+1}/{num_samples}] Loading cluster {cluster_id} | Grid: {grid_cell} | File: {os.path.basename(pq_path)}")
    
    # Read just the specific row group from the massive parquet file
    dataset = pq.ParquetFile(pq_path)
    
    # Read the RGB bands: B04 (Red), B03 (Green), B02 (Blue)
    table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
    
    # The columns contain bytes that decode into 16-bit PIL Images
    def bytes_to_array(b):
        if isinstance(b, dict):
            b = b['bytes']
        return np.array(Image.open(io.BytesIO(b))).astype(float)
        
    R = bytes_to_array(table.column('B04')[0].as_py())
    G = bytes_to_array(table.column('B03')[0].as_py())
    B = bytes_to_array(table.column('B02')[0].as_py())
    
    # Stack into an RGB numpy array
    rgb = np.dstack((R, G, B))
    
    # Simple brightness scaling for Sentinel-2 satellite imagery (typically values range 0-10000)
    # Divide by 3000 to normalize, clip to [0,1], then scale to [0,255] for standard PNG
    rgb = rgb / 3000.0
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    # Save image
    img = Image.fromarray(rgb)
    out_path = os.path.join(output_dir, f"cluster_{cluster_id}_{grid_cell}.png")
    img.save(out_path)
    print(f"  -> Saved to {out_path}")

print(f"\nDone! You can find the images in: {output_dir}")
