import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt

def bytes_to_array(b):
    if isinstance(b, dict):
        b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

def main():
    base_dir = "/data/databases/Core-S2L2A"
    print("Loading metadata...")
    manifest = pd.read_parquet(os.path.join(base_dir, "metadata.parquet"), columns=["grid_cell", "parquet_url", "parquet_row"])
    
    # Shuffle manifest
    manifest = manifest.sample(frac=1, random_state=42).reset_index(drop=True)

    found_count = 0
    max_to_find = 2
    
    for i, row in manifest.iterrows():
        if i % 1000 == 0:
            print(f"Checked {i} images...")
        parquet_file = os.path.join(base_dir, "images", row['parquet_url'].split("/")[-1])
        row_group = int(row['parquet_row'])
        
        try:
            dataset = pq.ParquetFile(parquet_file)
            table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
        except Exception as e:
            continue
            
        R = bytes_to_array(table.column('B04')[0].as_py())
        G = bytes_to_array(table.column('B03')[0].as_py())
        B = bytes_to_array(table.column('B02')[0].as_py())
        
        rgb = np.stack([R, G, B], axis=-1)
        zeros_pct = (rgb == 0).mean() * 100
        
        # Look for somewhat substantial missing data so it's visible, e.g., > 1% and < 50%
        if 1 < zeros_pct < 50:
            print(f"Found image with {zeros_pct:.2f}% missing data!")
            print(f"  Parquet: {parquet_file}, Row: {row_group}")
            
            rgb_viz = rgb / 3000.0
            rgb_viz = np.clip(rgb_viz, 0, 1)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb_viz)
            plt.title(f"Missing Data: {zeros_pct:.2f}%")
            plt.axis('off')
            
            out_name = f"missing_data_example_{found_count+1}.png"
            plt.savefig(out_name, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved visualization to {out_name}")
            
            found_count += 1
            if found_count >= max_to_find:
                break

if __name__ == '__main__':
    main()
