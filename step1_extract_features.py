"""
Step 1 – Feature Extraction
============================
Runs DINOv3-ViT-L/16 (3-channel, RGB) over **every** tile in the
cloud-filtered manifest and saves the CLS-token embeddings to disk.

Embedding all tiles (sample_fraction=1.0 in config) is required so that
Step 2 can assign exact cluster labels to every tile without any
proportional-extrapolation approximation.

The ViT-L/16 model produces 1024-dimensional CLS tokens.  Each embedding is
a compact semantic fingerprint of the tile that the k-means step will cluster.

Output
------
``<output_dir>/embeddings.npz``
  Arrays:
    embeddings  – float32, shape (N, 1024)
    indices     – int64,   shape (N,)   index into the manifest rows
    manifest_path – str, path to the manifest used
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _TileDataset(Dataset):
    """Lightweight dataset that reads tiles from the manifest on-the-fly.

    Returns a 3-channel (RGB) float32 tensor of shape (3, crop_size, crop_size)
    normalised to [0, 1], plus the manifest row index.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        rgb_band_indices: List[int],
        crop_size: int = 224,
    ) -> None:
        self.manifest        = manifest.reset_index(drop=True)
        self.rgb_band_indices = rgb_band_indices
        self.crop_size        = crop_size

        # Map band indices to band names (same order as CoreS2L2A)
        _BAND_NAMES = (
            "B01", "B02", "B03", "B04", "B08",
            "B05", "B06", "B07", "B8A", "B09",
            "B11", "B12",
        )
        self.rgb_col_names = [_BAND_NAMES[i] for i in rgb_band_indices]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.manifest)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row    = self.manifest.iloc[idx]
        fpath  = row["parquet_file"]
        rg_idx = int(row["row_group"])
        r_idx  = int(row["row_in_rg"])

        # Read only the 3 RGB columns for efficiency
        cols = list(self.rgb_col_names)
        try:
            table = pq.ParquetFile(fpath).read_row_group(rg_idx, columns=cols)
        except Exception as exc:
            log.warning("Cannot read %s rg=%d: %s – returning zeros", fpath, rg_idx, exc)
            return torch.zeros(3, self.crop_size, self.crop_size), idx

        bands: List[torch.Tensor] = []
        is_bad = False
        for col in cols:
            blob = table.column(col)[r_idx]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                blob = blob.as_py()
            if blob is None:
                is_bad = True
                break
            band = _decode_band(blob)
            bands.append(band)

        if is_bad:
            # We return a specially marked tensor or just zeros with a flag, 
            # but to keep it simple, we will return an empty tensor and handle in collation or extracting.
            # wait, the simplest way is to return the image and an indicator.
            return torch.zeros(3, self.crop_size, self.crop_size), -1

        img = torch.stack(bands, dim=0)   # (3, H, W)
        
        # Check zeros before scaling
        zero_mask = (img == 0).all(dim=0)
        zeros_pct = zero_mask.float().mean().item() * 100.0
        if zeros_pct > 45.0:
            return torch.zeros(3, self.crop_size, self.crop_size), -1

        img = img / 10_000.0
        img = torch.clamp(img, min=0.0, max=1.0)

        img = _resize_img(img, self.crop_size)
        return img, idx


# ---------------------------------------------------------------------------
# Band decoding helper (inline, no rasterio import at module level for workers)
# ---------------------------------------------------------------------------

def _decode_band(blob: bytes) -> torch.Tensor:
    from rasterio.io import MemoryFile
    import torch.nn.functional as F
    with MemoryFile(blob) as mf:
        with mf.open() as src:
            arr = src.read(1).astype(np.float32)   # (H, W)
    return torch.from_numpy(arr)  # (H, W)


def _resize_img(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a (C, H, W) tensor to (C, size, size) to preserve global context."""
    import torch.nn.functional as F
    img = F.interpolate(
        img.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False
    ).squeeze(0)
    return img


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(cfg: dict, device: torch.device) -> nn.Module:
    """Load DINOv3 ViT-S/16 from a local weights file and return in eval mode."""
    # Add the dinov3-testing-stuff repo to sys.path so we can import its hub
    repo_root = os.path.join(
        os.path.dirname(__file__), "..", "dinov3-testing-stuff"
    )
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    feat_cfg     = cfg["feature_extraction"]
    model_arch   = feat_cfg.get("model_arch", "vits16")
    weights_path = feat_cfg["weights_path"]

    log.info("Loading model arch=%s from %s", model_arch, weights_path)

    from dinov3.hub.backbones import dinov3_vits16, dinov3_vitl16

    arch_fn = {"vits16": dinov3_vits16, "vitl16": dinov3_vitl16}
    if model_arch not in arch_fn:
        raise ValueError(f"Unknown model_arch: {model_arch}. Choose vits16 or vitl16.")

    model = arch_fn[model_arch](pretrained=True, weights=weights_path)
    model = model.to(device).eval()
    log.info("Model loaded: embed_dim=%d", model.embed_dim)
    return model


# ---------------------------------------------------------------------------
# Normalisation constants (SAT-493M -style, used by DINOv3 RGB models)
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1)
_STD  = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1)


@torch.inference_mode()
def _extract_embeddings(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    total:    int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over all batches; return (embeddings, indices)."""
    all_embs:    List[np.ndarray] = []
    all_indices: List[np.ndarray] = []

    mean = _MEAN.to(device)
    std  = _STD.to(device)

    processed = 0
    for imgs, idxs in loader:
        # Filter out bad images (idx == -1)
        valid_mask = idxs != -1
        if not valid_mask.any():
            continue
            
        imgs = imgs[valid_mask]
        idxs = idxs[valid_mask]

        imgs = imgs.to(device, non_blocking=True)
        # ImageNet normalisation
        imgs = (imgs - mean) / std

        # Forward pass – use the CLS token (index 0)
        out = model(imgs)
        if isinstance(out, dict):
            embs = out["x_norm_clstoken"]   # (B, D)
        elif isinstance(out, (tuple, list)):
            embs = out[0]
        else:
            embs = out

        all_embs.append(embs.float().cpu().numpy())
        all_indices.append(idxs.numpy())

        processed += len(imgs)
        if processed % 5000 == 0 or processed >= total:
            log.info("  Extracted %d / %d embeddings", processed, total)

    embeddings = np.concatenate(all_embs, axis=0)   # (N', D)
    indices    = np.concatenate(all_indices, axis=0) # (N',)
    return embeddings, indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> str:
    """Run feature extraction and return the path to the embeddings .npz file."""
    data_cfg = cfg["data"]
    feat_cfg = cfg["feature_extraction"]

    output_dir      = data_cfg["output_dir"]
    
    # We use all tiles, reading straight from the raw metadata.parquet
    manifest_fname  = "full_dataset_manifest.parquet"
    manifest_path   = os.path.join(output_dir, manifest_fname)
    embeddings_fname = feat_cfg.get("embeddings_file", "embeddings.npz")
    embeddings_path  = os.path.join(output_dir, embeddings_fname)

    rgb_indices      = list(feat_cfg.get("rgb_band_indices", [3, 2, 1]))
    crop_size        = int(feat_cfg.get("crop_size", 224))
    batch_size       = int(feat_cfg.get("batch_size", 64))
    num_workers      = int(feat_cfg.get("num_workers", 8))
    device_str       = feat_cfg.get("device", "cuda")

    os.makedirs(output_dir, exist_ok=True)

    # Load manifest – ALL tiles are embedded
    if os.path.exists(manifest_path):
        log.info("Loading manifest from %s", manifest_path)
        manifest = pd.read_parquet(manifest_path)
    else:
        meta_path = os.path.join(data_cfg["images_dir"], "..", "metadata.parquet")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing dataset metadata file at {meta_path}."
            )
        log.info("Building full manifest from raw %s", meta_path)
        manifest = pd.read_parquet(meta_path, columns=["grid_cell", "parquet_url", "parquet_row"])
        
        def _to_local_path(hf_url: str) -> str:
            prefix, fname = hf_url.split("images/")
            return os.path.join(data_cfg["images_dir"], fname)
            
        manifest["parquet_file"] = manifest["parquet_url"].apply(_to_local_path)
        manifest["row_group"]    = manifest["parquet_row"].astype(int)
        manifest["row_in_rg"]    = 0
        
        manifest.to_parquet(manifest_path)
        log.info("Saved formatted full manifest to %s", manifest_path)

    n_tiles = len(manifest)
    log.info("Embedding ALL %d tiles (no sub-sampling).", n_tiles)

    # Device
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Model
    model = _load_model(cfg, device)

    # Dataset & loader – use the full manifest directly
    dataset = _TileDataset(manifest, rgb_indices, crop_size)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device_str == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    log.info("Extracting embeddings on %s …", device)
    embeddings, indices = _extract_embeddings(model, loader, device, len(dataset))

    # Save – indices are already direct manifest row indices (0 … N-1)
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        indices=indices,
        manifest_path=np.array(manifest_path),
    )
    log.info(
        "Saved %d embeddings (dim=%d) to %s",
        len(embeddings),
        embeddings.shape[1],
        embeddings_path,
    )
    return embeddings_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1: DINOv3-ViT-L/16 feature extraction for Core-S2L2A subset selection."
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    run(cfg)
