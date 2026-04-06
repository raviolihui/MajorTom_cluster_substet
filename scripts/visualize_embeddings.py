"""Simple exploratory tool for the embeddings produced in Step 1."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA


def load_embeddings(path: Path, max_samples: int, seed: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    if max_samples and len(embeddings) > max_samples:
        rng = np.random.default_rng(seed)
        choice = rng.choice(len(embeddings), size=max_samples, replace=False)
        embeddings = embeddings[choice]
    return embeddings


def plot_pca(
    embeddings: np.ndarray,
    output: Path,
    dpi: int,
) -> None:
    pca = PCA(n_components=4, random_state=42, svd_solver="randomized")
    projected = pca.fit_transform(embeddings)
    colors = (projected[:, 3] - projected[:, 3].min()) / (np.ptp(projected[:, 3]) + 1e-6)

    fig = plt.figure(figsize=(9, 6))
    ax: Axes = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        c=colors,
        cmap=cm.viridis,
        s=6,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title("Embeddings PCA (3D) – color = PCA component 4")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    fig.colorbar(scatter, label="PCA 4 (normalized)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    print(f"Saved PCA visualization to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Step 1 embeddings via PCA and scatter plot.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/embeddings.npz"),
        help="Path to the Step 1 embeddings archive.",
    )
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum random subset to plot.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sub-sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/embeddings_pca3d.png"),
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for the saved PNG.")
    args = parser.parse_args()

    embeddings = load_embeddings(args.embeddings, args.max_samples, args.seed)
    plot_pca(embeddings, args.output, args.dpi)


if __name__ == "__main__":
    main()
