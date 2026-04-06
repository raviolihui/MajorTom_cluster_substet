import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Import step1 for dataset and model loader
import step1_extract_features as s1

def main():
    print("Loading config...")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Use CPU by default to avoid competing with step1 if it's running, 
    # but try CUDA if it's free. We'll stick to cuda if available for speed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load manifest and sample mathematically significant number of tiles (e.g. 2500)
    meta_path = os.path.join(cfg["data"]["images_dir"], "..", "metadata.parquet")
    manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])
    
    n_sample = 2500 # Just enough to calculate a 1024-D PCA cleanly
    print(f"Sampling {n_sample} tiles for PCA variance analysis...")
    manifest = manifest.sample(n=n_sample, random_state=42)

    def _to_local_path(hf_url: str) -> str:
        prefix, fname = hf_url.split("images/")
        return os.path.join("/data/databases/Core-S2L2A/images", fname)

    manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
    manifest["row_group"]    = manifest["parquet_row"].astype(int)
    manifest["row_in_rg"]    = 0

    # 2. Extract features
    print("Loading model...")
    model = s1._load_model(cfg, device)
    rgb_indices = cfg["feature_extraction"].get("rgb_band_indices", [3, 2, 1])
    crop_size = cfg["feature_extraction"].get("crop_size", 224)
    
    dataset = s1._TileDataset(manifest, rgb_indices, crop_size)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    print("Extracting embeddings for PCA sample...")
    embeddings, _ = s1._extract_embeddings(model, loader, device, len(dataset))

    # 3. Fit PCA and compute explained variance
    print("Fitting PCA...")
    max_components = min(embeddings.shape[1], n_sample)
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
    pca.fit(embeddings)

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), cumsum, lw=2, color='#1f77b4')
    
    # Highlight the 128 component mark
    if max_components >= 128:
        val_128 = cumsum[127]
        plt.axvline(x=128, color='red', linestyle='--', label=f'128 Comps ({val_128:.1%} Var)')
        plt.plot(128, val_128, 'ro') # red dot
    
    # Add arbitrary reference bands for 90% and 95%
    plt.axhline(y=0.90, color='green', linestyle=':', label='90% Variance')
    plt.axhline(y=0.95, color='orange', linestyle=':', label='95% Variance')
    
    plt.title('Cumulative Explained Variance by PCA Components (DINOv3-ViT-L)')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_file = "pca_variance_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Success! Plot saved to {out_file}")

if __name__ == "__main__":
    # Ensure Make_MajorTom_sbst is in sys.path so we can import step1_extract_features
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
