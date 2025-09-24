#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    DEFAULT_X_METRIC, collect_metric_data, compute_stats, configure_scientific_x, 
    plot_mean_with_shade, build_label_map, logger, add_start_point
)

def plot_curvature_clip_figure(run_paths: List[str], x_metric: str, output_dir: str, 
                              ema_alpha: float | None, ci_type: str, label_map: Dict[str, str]):
    """Create a single plot with all 3 curvature clip ratio metrics."""
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    
    # Define the three metrics to plot
    metrics = [
        ("train/curvature_clip_ratio_total_full", "Total"),
        # ("train/curvature_clip_ratio_total_fisher", "Fisher"), 
        # ("train/curvature_clip_ratio_total_hessian", "Hessian")
    ]
    
    # Define the desired order for consistent colors
    desired_order = ["GRPO", "GRPO (A)", "CAPO", "REINFORCE", "DrGRPO"]
    
    # Plot each metric for each run
    for metric_name, metric_label in metrics:
        for rp in run_paths:
            df = collect_metric_data(rp, x_metric, metric_name)
            df = add_start_point(df, x_metric, y_metric=metric_name, mode="avg")
            if df is None:
                logger.warning(f"[{metric_name}] Metric not found or could not align in {rp}: {metric_name}")
                continue
            
            stats = compute_stats(df, x_metric, ema_alpha, ci_type)
            # If single run path, use only metric label; otherwise include method name
            if len(run_paths) == 1:
                label = metric_label
            else:
                label = f"{label_map.get(rp)} ({metric_label})"
            plot_mean_with_shade(ax, stats[x_metric], stats["mean"], stats["lo"], stats["hi"], label)


    ax.set_xlabel("Training Completions", fontsize=16)
    ax.set_title("Token Rejection Rate")
    ax.set_xlim(0, 10000)
    
    # Add legend
    # ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=True)
    
    ax.grid(True)
    ax.set_xticks(np.arange(0, 10000, 2000))
    # Reduce number of y-axis ticks
    ax.locator_params(axis='y', nbins=5)
    configure_scientific_x(ax)
    
    # Adjust layout and save
    fig.tight_layout()
    
    # Save outputs (PNG + high-res PDF)
    out = Path(output_dir) / "curvature_clip_ratios.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    
    out_pdf = Path(output_dir) / "curvature_clip_ratios.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths", type=str, required=True,
                       help="Comma-separated list of run paths")
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--ema_alpha", type=float, default=1.0)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels", type=str, default="",
                       help="Comma-separated labels for experiments")
    args = parser.parse_args()

    run_paths = [p.strip() for p in args.run_paths.split(",") if p.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map = build_label_map(run_paths, args.labels)

    plot_curvature_clip_figure(run_paths, args.x_metric, args.output_dir, 
                              ema_alpha, args.ci_type, label_map)

if __name__ == "__main__":
    main()
