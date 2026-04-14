#!/bin/bash
#SBATCH --job-name=majortom_subset
#SBATCH --partition=gpu
#SBATCH --output=logs/majortom_subset_%j.log
#SBATCH --error=logs/majortom_subset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=50:00:00
# Adjust partition / account / qos to your cluster



# ─── Environment ────────────────────────────────────────────────────────────
source /opt/ohpc/pub/miniforge3/etc/profile.d/conda.sh
conda activate envr

# ─── Paths ──────────────────────────────────────────────────────────────────
#go to script directory make_makortom_sbts where run_pipeline.py is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/carmenoliver/my_projects/Make_MajorTom_sbst #"$SCRIPT_DIR"
mkdir -p logs

# ─── Pipeline ───────────────────────────────────────────────────────────────
# Step 1 = feature extraction (embed all tiles with ViT-L/16)
# Step 2 = k-means balanced sampling → subset_manifest.parquet
#
# To resume from k-means only (embeddings already computed):
#   python run_pipeline.py --config config.yaml --steps 2

echo "Starting pipeline at $(date)"
python run_pipeline.py --config config.yaml --steps 1,2
STATUS=$?

echo "Pipeline finished at $(date) with exit code $STATUS"
exit $STATUS
