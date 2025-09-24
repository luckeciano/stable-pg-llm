#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    DEFAULT_X_METRIC, TRAIN_METRIC,
    collect_metric_data, compute_stats, configure_scientific_x, 
    plot_mean_with_shade, build_label_map, logger, add_start_point
)

def plot_first_subplot(ax, run_paths: List[str], x_metric: str,
                      ema_alpha: float | None, ci_type: str, label_map: Dict[str, str], title: str):
    """Plot first subplot with custom settings."""
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, x_metric, TRAIN_METRIC)
        df = add_start_point(df, x_metric, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping first plot.")
        return

    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # CUSTOMIZE FIRST PLOT HERE
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.6, 0.83)
    ax.set_xlabel("Training Completions")
    ax.set_ylabel("MATH Accuracy")
    ax.set_title(title)
    
    ax.legend(bbox_to_anchor=(0.98, 0.02), loc="lower right", ncol=1, fontsize=12)
    ax.grid(True)
    ax.set_xticks(np.arange(0, 12000, 2000))
    ax.set_yticks(np.arange(0.6, 0.825, 0.05))
    configure_scientific_x(ax)

def plot_second_subplot(ax, run_paths: List[str], x_metric: str,
                       ema_alpha: float | None, ci_type: str, label_map: Dict[str, str], title: str):
    """Plot second subplot with custom settings."""
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, x_metric, TRAIN_METRIC)
        df = add_start_point(df, x_metric, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping second plot.")
        return

    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # CUSTOMIZE SECOND PLOT HERE
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.60, 0.80)
    ax.set_xlabel("Training Completions")
    ax.set_title(title)
    
    ax.legend(bbox_to_anchor=(0.98, 0.02), loc="lower right", ncol=1, fontsize=12)
    ax.grid(True)
    ax.set_xticks(np.arange(0, 12000, 2000))
    ax.set_yticks(np.arange(0.60, 0.81, 0.05))
    configure_scientific_x(ax)

def plot_third_subplot(ax, run_paths: List[str], x_metric: str,
                      ema_alpha: float | None, ci_type: str, label_map: Dict[str, str], title: str):
    """Plot third subplot with custom settings."""
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, x_metric, TRAIN_METRIC)
        df = add_start_point(df, x_metric, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping third plot.")
        return

    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # CUSTOMIZE THIRD PLOT HERE
    ax.set_xlim(0, 10000)
    ax.set_ylim(0.5, 0.8)
    ax.set_xlabel("Training Completions")
    ax.set_title(title)
    
    ax.legend(bbox_to_anchor=(0.98, 0.02), loc="lower right", ncol=1, fontsize=12)
    ax.grid(True)
    ax.set_xticks(np.arange(0, 12000, 2000))
    ax.set_yticks(np.arange(0.5, 0.81, 0.05))
    configure_scientific_x(ax)

def plot_three_ablation(run_paths_1: List[str], run_paths_2: List[str], run_paths_3: List[str], 
                       x_metric: str, output_dir: str, ema_alpha: float | None, ci_type: str, 
                       label_map_1: Dict[str, str], label_map_2: Dict[str, str], label_map_3: Dict[str, str],
                       title_1: str, title_2: str, title_3: str):
    """Create a combined figure with three train metric plots."""
    
    # Create figure with 3 subplots (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot first set of runs on first subplot
    plot_first_subplot(ax1, run_paths_1, x_metric, ema_alpha, ci_type, label_map_1, title_1)
    
    # Plot second set of runs on second subplot
    plot_second_subplot(ax2, run_paths_2, x_metric, ema_alpha, ci_type, label_map_2, title_2)
    
    # Plot third set of runs on third subplot
    plot_third_subplot(ax3, run_paths_3, x_metric, ema_alpha, ci_type, label_map_3, title_3)
    
    # Adjust layout and save
    fig.tight_layout()
    
    # Save outputs (PNG + high-res PDF)
    out = Path(output_dir) / "ablation_optimizer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    
    out_pdf = Path(output_dir) / "ablation_optimizer.pdf"
    fig.savefig(out_pdf, dpi=300)
    
    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths_1", type=str, required=True, 
                       help="Comma-separated list of run paths for first plot")
    parser.add_argument("--run_paths_2", type=str, required=True,
                       help="Comma-separated list of run paths for second plot")
    parser.add_argument("--run_paths_3", type=str, required=True,
                       help="Comma-separated list of run paths for third plot")
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--output_dir", type=str, default="plots/three_ablation")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels_1", type=str, default="",
                       help="Comma-separated labels for first plot runs")
    parser.add_argument("--labels_2", type=str, default="",
                       help="Comma-separated labels for second plot runs")
    parser.add_argument("--labels_3", type=str, default="",
                       help="Comma-separated labels for third plot runs")
    parser.add_argument("--title_1", type=str, default="Ablation 1",
                       help="Title for first plot")
    parser.add_argument("--title_2", type=str, default="Ablation 2",
                       help="Title for second plot")
    parser.add_argument("--title_3", type=str, default="Ablation 3",
                       help="Title for third plot")
    args = parser.parse_args()

    run_paths_1 = [p.strip() for p in args.run_paths_1.split(",") if p.strip()]
    run_paths_2 = [p.strip() for p in args.run_paths_2.split(",") if p.strip()]
    run_paths_3 = [p.strip() for p in args.run_paths_3.split(",") if p.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map_1 = build_label_map(run_paths_1, args.labels_1)
    label_map_2 = build_label_map(run_paths_2, args.labels_2)
    label_map_3 = build_label_map(run_paths_3, args.labels_3)

    plot_three_ablation(run_paths_1, run_paths_2, run_paths_3, args.x_metric, args.output_dir, 
                       ema_alpha, args.ci_type, label_map_1, label_map_2, label_map_3,
                       args.title_1, args.title_2, args.title_3)

if __name__ == "__main__":
    main()
