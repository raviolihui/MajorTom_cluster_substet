import argparse
import logging
import os
import yaml
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def _to_local_path(hf_url: str) -> str:
    """Map huggingface Hub URL to local Core-S2L2A/images file."""
    prefix, fname = hf_url.split("images/")
    return os.path.join("/data/databases/Core-S2L2A/images", fname)

def _decode_band(blob: bytes) -> torch.Tensor:
    from rasterio.io import MemoryFile
    with MemoryFile(blob) as mf:
        with mf.open() as src:
            arr = src.read(1).astype(np.float32)
    return torch.from_numpy(arr)

class FastS2Dataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, rgb_indices: List[int]):
        self.manifest = manifest.to_dict("records")
        self.rgb_indices = rgb_indices
        _BAND_NAMES = ("B01", "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B09", "B11", "B12"); self.cols = [_BAND_NAMES[i] for i in rgb_indices]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest[idx]
        fpath = row["parquet_file"]
        rg_idx = row["row_group"]
        r_idx = row["row_in_rg"]

        try:
            table = pq.ParquetFile(fpath).read_row_group(rg_idx, columns=self.cols)
            bands = []
            for col in self.cols:
                blob = table.column(col)[r_idx]
                if not isinstance(blob, (bytes, bytearray, memoryview)):
                    blob = blob.as_py()
                band = _decode_band(blob)
                bands.append(band)
            img = torch.stack(bands, dim=0)   # (3, H, W)
            # Apply typical optical scaling
            img = img / 10_000.0
            return img, True
        except Exception as exc:
            # log.warning(f"Error reading row {idx}: {exc}")
            return torch.zeros(3, 1068, 1068), False

def compute_stats(manifest_df, rgb_indices=[3, 2, 1], batch_size=64, num_workers=16):
    dataset = FastS2Dataset(manifest_df, rgb_indices)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    total_pixels = 0
    # Store sum and squared sum over the whole dataset
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_sq_sum = torch.zeros(3, dtype=torch.float64)
    
    log.info(f"Starting stats computation over {len(manifest_df)} tiles...")
    
    visited = 0
    valid_tiles = 0
    for imgs, valids in loader:
        # imgs shape: (B, 3, H, W)
        valid_mask = valids == True
        if not valid_mask.any():
            continue
            
        valid_imgs = imgs[valid_mask]
        
        # Calculate sum and sum of squares for this chunk
        # Suming over spatial dimensions (B, 3, H, W) -> (3,)
        b_sum = valid_imgs.sum(dim=[0, 2, 3])
        b_sq_sum = (valid_imgs ** 2).sum(dim=[0, 2, 3])
        
        n_pixels_in_batch = valid_imgs.shape[0] * valid_imgs.shape[2] * valid_imgs.shape[3]
        
        pixel_sum += b_sum.double()
        pixel_sq_sum += b_sq_sum.double()
        total_pixels += n_pixels_in_batch
        valid_tiles += valid_imgs.shape[0]
        visited += imgs.shape[0]
        
        if visited % 1024 == 0:
            log.info(f"Processed {visited} / {len(manifest_df)} tiles...")

    msg = f"Finished processing! Computed stats over {valid_tiles} valid tiles."
    log.info(msg)
    
    mean = pixel_sum / total_pixels
    variance = (pixel_sq_sum / total_pixels) - (mean ** 2)
    std = torch.sqrt(variance)
    
    return mean.float().tolist(), std.float().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample_fraction", type=float, default=0.01)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    meta_path = os.path.join(cfg["data"]["images_dir"], "..", "metadata.parquet")
    log.info(f"Loading manifest from {meta_path}...")
    manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])

    if args.sample_fraction < 1.0:
        n_sample = int(len(manifest) * args.sample_fraction)
        log.info(f"Sub-sampling {args.sample_fraction*100}% of the dataset ({n_sample} tiles) for stats approximation...")
        manifest = manifest.sample(n=n_sample, random_state=42)
    
    manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
    manifest["row_group"]    = manifest["parquet_row"].astype(int)
    manifest["row_in_rg"]    = 0

    rgb_indices = cfg["feature_extraction"]["rgb_band_indices"]
    
    mean, std = compute_stats(manifest, rgb_indices, batch_size=8, num_workers=8)
    
    log.info("===" * 10)
    log.info(f"COMPUTED S2L2A DATASET STATISTICS:")
    log.info(f"MEAN: {mean}")
    log.info(f"STD:  {std}")
    log.info("===" * 10)
    
    # Optional: Write them out or update step1 directly?
    print(f"\nYou should paste these exactly into step1_extract_features.py:")
    print(f"_MEAN = torch.tensor([{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]).view(3, 1, 1)")
    print(f"_STD  = torch.tensor([{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]).view(3, 1, 1)")
