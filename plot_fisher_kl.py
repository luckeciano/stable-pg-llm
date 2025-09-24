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

def plot_combined_metric_subplot(ax, run_paths_no_mask: List[str], run_paths_mask: List[str], 
                                x_metric: str, metric_name_no_mask: str, metric_name_mask: str,
                                ema_alpha: float | None, ci_type: str, 
                                label_map_no_mask: Dict[str, str], label_map_mask: Dict[str, str], 
                                title: str, ylim: tuple = None, yticks: List[float] = None, 
                                y_nbins: int = 4):
    """Plot combined metric (both masked and non-masked) on the given subplot axis."""
    stats_by_label = {}
    
    # Process non-masked runs
    for rp in run_paths_no_mask:
        df = collect_metric_data(rp, x_metric, metric_name_no_mask)
        df = add_start_point(df, x_metric, y_metric=metric_name_no_mask, mode="avg")
        if df is None:
            logger.warning(f"[{metric_name_no_mask}] Metric not found or could not align in {rp}: {metric_name_no_mask}")
            continue
        stats_by_label[label_map_no_mask.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)
    
    # Process masked runs
    for rp in run_paths_mask:
        df = collect_metric_data(rp, x_metric, metric_name_mask)
        df = add_start_point(df, x_metric, y_metric=metric_name_mask, mode="avg")
        if df is None:
            logger.warning(f"[{metric_name_mask}] Metric not found or could not align in {rp}: {metric_name_mask}")
            continue
        stats_by_label[label_map_mask.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning(f"No data found for {title} across all run_paths. Skipping {title} plot.")
        return

    # Define the desired order for consistent colors
    desired_order = ["GRPO", "GRPO (A)", "CAPO", "DrGRPO", "REINFORCE"]
    
    # Plot in the desired order
    for label in desired_order:
        if label in stats_by_label:
            st = stats_by_label[label]
            plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)
    
    # Plot any remaining labels not in the desired order
    for label, st in stats_by_label.items():
        if label not in desired_order:
            plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # Add reference line for GRPO max value
    if "GRPO" in stats_by_label:
        grpo_stats = stats_by_label["GRPO"]
        grpo_max = grpo_stats["mean"].max()
        blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        ax.axhline(grpo_max, linestyle="--", color=blue, label="GRPO (max)")

    ax.set_xlabel("Training Completions", fontsize=18)
    ax.set_title(title)
    ax.set_xlim(0, 10000)
    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
    
    # Don't add individual legends - will be handled at figure level
    ax.grid(True)
    ax.set_xticks(np.arange(0, 10000, 2000))
    # Set y-axis to specified number of ticks
    ax.locator_params(axis='y', nbins=y_nbins)
    configure_scientific_x(ax)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


def plot_combined_figure(run_paths_no_mask: List[str], run_paths_mask: List[str], 
                        x_metric: str, output_dir: str, ema_alpha: float | None, 
                        ci_type: str, label_map_no_mask: Dict[str, str], 
                        label_map_mask: Dict[str, str]):
    """Create a single 1x4 figure with all 4 plots."""
    
    # Set font sizes
    plt.rcParams.update({'font.size': 20})
    
    # Create figure with 1x4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    
    # Plot 1: Token fisher KL divergence (both masked and non-masked)
    plot_combined_metric_subplot(
        ax1, run_paths_no_mask, run_paths_mask, x_metric, 
        "train/token_fisher_kl_divergence", "train/masked_token_fisher_kl_divergence",
        ema_alpha, ci_type, label_map_no_mask, label_map_mask, "$m_{F}$ (Token)",
        ylim=(0, 0.01)
    )
    
    # Plot 2: Global fisher KL divergence (both masked and non-masked)
    plot_combined_metric_subplot(
        ax2, run_paths_no_mask, run_paths_mask, x_metric, 
        "train/global_fisher_kl_divergence", "train/masked_global_fisher_kl_divergence",
        ema_alpha, ci_type, label_map_no_mask, label_map_mask, "$m_{F}$ (Global)",
        ylim=(0, 2e-5), y_nbins=5
    )
    
    # Plot 3: Token full update term (both masked and non-masked)
    plot_combined_metric_subplot(
        ax3, run_paths_no_mask, run_paths_mask, x_metric, 
        "train/token_full_update_term", "train/masked_token_full_update_term",
        ema_alpha, ci_type, label_map_no_mask, label_map_mask, "$m_{H}$ (Token)",
        ylim=(0, 0.03)
    )
    
    # Plot 4: Global full update term (both masked and non-masked)
    plot_combined_metric_subplot(
        ax4, run_paths_no_mask, run_paths_mask, x_metric, 
        "train/global_full_update_term", "train/masked_global_full_update_term",
        ema_alpha, ci_type, label_map_no_mask, label_map_mask, "$m_{H}$ (Global)",
        ylim=(0, 1.0)
    )
    
    # Create a single legend in the upper right corner of the second plot
    # Get handles and labels from the first subplot (they should be the same across all)
    handles, labels = ax1.get_legend_handles_labels()
    
    # Add the legend to the second subplot (ax2) in the upper right corner
    ax2.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
               ncol=1, frameon=True)
    
    # Adjust layout and save
    fig.tight_layout()
    
    # Save outputs (PNG + high-res PDF)
    out = Path(output_dir) / "fisher_kl_and_full_update.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    
    out_pdf = Path(output_dir) / "fisher_kl_and_full_update.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths_no_mask", type=str, required=True,
                       help="Comma-separated list of run paths for non-masked experiments")
    parser.add_argument("--run_paths_mask", type=str, required=True,
                       help="Comma-separated list of run paths for masked experiments")
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--ema_alpha", type=float, default=0.35)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels_no_mask", type=str, default="",
                       help="Comma-separated labels for non-masked experiments")
    parser.add_argument("--labels_mask", type=str, default="",
                       help="Comma-separated labels for masked experiments")
    args = parser.parse_args()

    run_paths_no_mask = [p.strip() for p in args.run_paths_no_mask.split(",") if p.strip()]
    run_paths_mask = [p.strip() for p in args.run_paths_mask.split(",") if p.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map_no_mask = build_label_map(run_paths_no_mask, args.labels_no_mask)
    label_map_mask = build_label_map(run_paths_mask, args.labels_mask)

    plot_combined_figure(run_paths_no_mask, run_paths_mask, args.x_metric, 
                        args.output_dir, ema_alpha, args.ci_type, 
                        label_map_no_mask, label_map_mask)

if __name__ == "__main__":
    main()
