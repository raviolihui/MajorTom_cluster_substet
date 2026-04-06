import pyarrow.parquet as pq
import glob
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt

def bytes_to_array(b):
    if b is None:
        return np.zeros((256, 256))
    if isinstance(b, dict):
        b = b['bytes']
    return np.array(Image.open(io.BytesIO(b))).astype(float)

def main():
    base_dir = "/data/databases/Core-S2L2A/images"
    parquet_files = sorted(glob.glob(os.path.join(base_dir, "*.parquet")))
    np.random.seed(42)
    np.random.shuffle(parquet_files)
    
    found_count = 0
    max_to_find = 2
    
    for parquet_file in parquet_files:
        print(f"Reading {parquet_file}...")
        try:
            dataset = pq.ParquetFile(parquet_file)
            for row_group in range(dataset.num_row_groups):
                table = dataset.read_row_group(row_group, columns=['B04', 'B03', 'B02'])
                
                # Check all rows in this group
                for i in range(table.num_rows):
                    r_col = table.column('B04')[i].as_py()
                    if r_col is None:
                        continue
                        
                    R = bytes_to_array(table.column('B04')[i].as_py())
                    G = bytes_to_array(table.column('B03')[i].as_py())
                    B = bytes_to_array(table.column('B02')[i].as_py())
                    
                    rgb = np.stack([R, G, B], axis=-1)
                    zeros_pct = (rgb == 0).mean() * 100
                    
                    if 1 < zeros_pct < 50:
                        print(f"Found image with {zeros_pct:.2f}% missing data!")
                        print(f"  Location: {parquet_file}, Row Group: {row_group}, Row: {i}")
                        
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
                            return
        except Exception as e:
            print(f"Failed to process {parquet_file}: {e}")

if __name__ == '__main__':
    main()
