"""
run_pipeline.py – Master orchestrator
======================================
Runs the full Core-S2L2A subset-selection pipeline end-to-end:

  Step 1 │ Feature Extraction   (step1_extract_features.py)
  Step 2 │ k-Means Balancing    (step2_kmeans_faiss.py)

Usage
-----
    python run_pipeline.py [--config config.yaml] [--steps 1,2]

Examples
--------
    # Run all steps with the default config
    python run_pipeline.py

    # Run only extract_features
    python run_pipeline.py --steps 1

    # Run feature-extraction + balancing (cloud filtering already done)
    python run_pipeline.py --steps 1,2

    # Use a custom config
    python run_pipeline.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _run_step1(cfg: dict) -> None:
    from step1_extract_features import run as _step_run
    log.info("=" * 60)
    log.info("STEP 1 – Feature Extraction")
    log.info("=" * 60)
    t0 = time.perf_counter()
    out = _step_run(cfg)
    log.info("Step 1 finished in %.1f s → %s", time.perf_counter() - t0, out)


def _run_step2(cfg: dict) -> None:
    from step2_kmeans_faiss import run as _step_run
    log.info("=" * 60)
    log.info("STEP 2 – FAISS Spherical k-Means Balancing")
    log.info("=" * 60)
    t0 = time.perf_counter()
    out = _step_run(cfg)
    log.info("Step 2 finished in %.1f s → %s", time.perf_counter() - t0, out)


_STEP_MAP = {1: _run_step1, 2: _run_step2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, steps: List[int]) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Ensure the pipeline scripts directory is on the path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Ensure dinov3-testing-stuff is importable (needed by step 2)
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "dinov3-testing-stuff"))
    if os.path.isdir(repo_root) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    log.info("Config: %s", config_path)
    log.info("Output dir: %s", cfg["data"]["output_dir"])
    log.info("Running steps: %s", steps)

    t_total = time.perf_counter()
    for step in steps:
        if step not in _STEP_MAP:
            raise ValueError(f"Unknown step: {step}. Must be one of {list(_STEP_MAP)}")
        _STEP_MAP[step](cfg)

    log.info("=" * 60)
    log.info("Pipeline complete in %.1f s", time.perf_counter() - t_total)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Core-S2L2A subset-selection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml next to this script).",
    )
    p.add_argument(
        "--steps",
        default="1,2",
        help="Comma-separated list of steps to run (default: 1,2).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    steps  = [int(s.strip()) for s in args.steps.split(",")]

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    main(config_path, steps)
