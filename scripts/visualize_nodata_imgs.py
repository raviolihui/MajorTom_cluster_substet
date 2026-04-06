import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt

def decode_band(blob):
    """Decodes the binary byte blob from MajorTom into a numpy array."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        blob = blob.as_py()
    with MemoryFile(blob) as mf:
        with mf.open() as src:
            arr = src.read(1).astype(np.float32)
    return arr

def main():
    metadata_path = '/data/databases/MajorTom5T/outputs_filtered/images/metadata_full.parquet'
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    
    # Filter the exact 3 images
    high_nodata_df = df[df['nodata'] > 0.45]
    print(f"Found {len(high_nodata_df)} images with >45% nodata. Extracting RGB pixels...")

    os.makedirs('pics', exist_ok=True)

    for i, (idx, row) in enumerate(high_nodata_df.iterrows()):
        fpath = row["parquet_file"]
        rg_idx = int(row["row_group"])
        r_idx = int(row["row_in_rg"])
        grid_cell = row["grid_cell"]
        
        print(f"[{i+1}/3] Loading {grid_cell} from {os.path.basename(fpath)} (RowGroup: {rg_idx})")
        
        # Read the Red, Green, Blue bands (B04, B03, B02 in Sentinel-2)
        try:
            pf = pq.ParquetFile(fpath)
            # Handle the case where the subset extraction merged all into one row_group
            if pf.num_row_groups == 1 and rg_idx >= 1:
                r_idx = rg_idx
                rg_idx = 0

            table = pf.read_row_group(rg_idx, columns=['B04', 'B03', 'B02'])
        except Exception as e:
            print(f"Failed to read parquet: {e}")
            continue
            
        r_blob = table.column('B04')[r_idx]
        g_blob = table.column('B03')[r_idx]
        b_blob = table.column('B02')[r_idx]
        
        # Decode the blobs into raw numpy arrays (H, W)
        R = decode_band(r_blob)
        G = decode_band(g_blob)
        B = decode_band(b_blob)
        
        # Stack into (H, W, 3) 
        rgb = np.stack([R, G, B], axis=-1)
        
        # Sentinel-2 values are typically reflectance * 10_000. 
        # To make it visually bright to the human eye, we often divide by ~3000 and clip to [0, 1].
        # Anything above 3000 will simply appear as bright white.
        rgb_display = rgb / 3000.0
        rgb_display = np.clip(rgb_display, 0.0, 1.0)
        
        out_filename = f"pics/nodata_img_{i+1}_{grid_cell}.png"
        plt.imsave(out_filename, rgb_display)
        print(f"    -> Saved RGB visualization to {out_filename}")

if __name__ == "__main__":
    main()