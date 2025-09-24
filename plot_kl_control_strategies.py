#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

from plot_utils import (
    TRAIN_METRIC, compute_stats, configure_scientific_x, 
    plot_mean_with_shade, build_label_map, logger
)

# =========================
# Constants
# =========================
X_METRIC = "train/global_step"
FIXED_STARTPOINT_PATH = "plot_data/Qwen-2.5-7B-GRPO-NoBaseline-v2-NoLR"

# =========================
# Custom Functions for Fixed X_METRIC
# =========================
def collect_metric_data(run_path: str, y_metric: str) -> pd.DataFrame | None:
    """Collect metric data aligned on global_step for a single y_metric."""
    csv_files = glob.glob(os.path.join(run_path, "*", "metrics.csv"))
    series_list = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
            continue

        # Check if both required columns exist
        if "train/global_step" not in df.columns or y_metric not in df.columns:
            continue

        # Extract and clean data for this seed
        seed_data = df[["train/global_step", y_metric]].dropna()
        if seed_data.empty:
            continue

        run_id = Path(csv_file).parent.name
        seed_data = seed_data.rename(columns={y_metric: run_id})
        series_list.append(seed_data)

    if not series_list:
        return None

    # Merge all seeds on global_step
    result = series_list[0]
    for seed_data in series_list[1:]:
        result = pd.merge(result, seed_data, on="train/global_step", how="outer")

    return result.sort_values("train/global_step").reset_index(drop=True)

def _sorted_seed_dirs(base: str) -> List[Path]:
    """Get sorted list of seed directories."""
    dirs = [p for p in Path(base).glob("*") if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name)

def _value_at_step(df: pd.DataFrame, col: str, step: int) -> Optional[float]:
    """Return the last non-NaN value in col at global_step==step."""
    if X_METRIC not in df.columns or col not in df.columns:
        return None
    rows = df.loc[(df[X_METRIC] == step) & df[col].notna(), col]
    if rows.empty:
        return None
    return float(rows.iloc[-1])

def _collect_fixed_seed_values(fixed_path: str, step: int, y_metric: str) -> List[Optional[float]]:
    """Collect per-seed values from fixed path at given step."""
    vals: List[Optional[float]] = []
    for sd in _sorted_seed_dirs(fixed_path):
        csv_files = list(sd.glob("metrics.csv"))
        if not csv_files:
            vals.append(None)
            continue
        try:
            df_seed = pd.read_csv(csv_files[0])
        except Exception as e:
            logger.warning(f"[add_start_point] Could not read {csv_files[0]}: {e}")
            vals.append(None)
            continue

        v = _value_at_step(df_seed, y_metric, step)
        vals.append(v)
    return vals

def add_start_point(df: pd.DataFrame | None, fixed_path: str = FIXED_STARTPOINT_PATH,
                   y_metric: str | None = None, step: int = 10,
                   mode: Literal["per-seed", "avg"] = "avg") -> pd.DataFrame | None:
    """Add synthetic start point at x=0 using values from fixed path."""
    if df is None:
        return None

    run_cols = [c for c in df.columns if c != X_METRIC]
    if not run_cols:
        return df
    run_cols = sorted(run_cols)

    per_seed_vals = _collect_fixed_seed_values(fixed_path, step, y_metric or TRAIN_METRIC)
    if not per_seed_vals:
        logger.warning(f"[add_start_point] No seed dirs found under fixed_path: {fixed_path}")
        return df

    start_row = {X_METRIC: 0.0}

    if mode == "avg":
        arr = pd.Series(per_seed_vals, dtype="float64")
        avg_val = float(arr.mean(skipna=True)) if arr.notna().any() else np.nan
        for col in run_cols:
            start_row[col] = avg_val
    else:
        k = len(run_cols)
        vals = per_seed_vals[:k]
        for i, col in enumerate(run_cols):
            start_row[col] = vals[i] if i < len(vals) else np.nan

    start_df = pd.DataFrame([start_row], columns=[X_METRIC] + run_cols)
    out = pd.concat([start_df, df], ignore_index=True).sort_values(X_METRIC).reset_index(drop=True)
    return out

def plot_first_subplot(ax, run_paths: List[str], ema_alpha: float | None, 
                      ci_type: str, label_map: Dict[str, str], title: str):
    """Plot first subplot with custom settings."""
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, TRAIN_METRIC)
        df = add_start_point(df, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, X_METRIC, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping first plot.")
        return

    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[X_METRIC], st["mean"], st["lo"], st["hi"], label)

    # CUSTOMIZE FIRST PLOT HERE
    ax.set_xlim(0, 102)
    ax.set_ylim(0.6, 0.80)
    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("MATH Accuracy")
    ax.set_title(title)
    
    # Legend will be created at the figure level
    ax.grid(True)
    ax.set_xticks(np.arange(0, 120, 20))
    ax.set_yticks(np.arange(0.6, 0.825, 0.05))
    configure_scientific_x(ax)

def plot_second_subplot(ax, run_paths: List[str], ema_alpha: float | None, 
                       ci_type: str, label_map: Dict[str, str], title: str):
    """Plot second subplot with custom settings."""
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, TRAIN_METRIC)
        df = add_start_point(df, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, X_METRIC, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping second plot.")
        return

    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[X_METRIC], st["mean"], st["lo"], st["hi"], label)

    # CUSTOMIZE SECOND PLOT HERE
    ax.set_xlim(0, 102)
    ax.set_ylim(0.60, 0.80)
    ax.set_xlabel("Gradient Steps")
    ax.set_title(title)
    
    # No legend for second subplot - using single legend for entire figure
    ax.grid(True)
    ax.set_xticks(np.arange(0, 120, 20))
    ax.set_yticks(np.arange(0.60, 0.81, 0.05))
    configure_scientific_x(ax)

def plot_two_kl_control(run_paths_1: List[str], run_paths_2: List[str], 
                       output_dir: str, ema_alpha: float | None, ci_type: str, 
                       label_map_1: Dict[str, str], label_map_2: Dict[str, str],
                       title_1: str, title_2: str):
    """Create a combined figure with two train metric plots."""
    
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot first set of runs on first subplot
    plot_first_subplot(ax1, run_paths_1, ema_alpha, ci_type, label_map_1, title_1)
    
    # Plot second set of runs on second subplot
    plot_second_subplot(ax2, run_paths_2, ema_alpha, ci_type, label_map_2, title_2)
    
    # Adjust layout first to make room for legend
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # Add more space for legend at bottom
    
    # Create single centralized legend using handles and labels from first subplot
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.01), loc="lower center", ncol=5)
    
    # Save outputs (PNG + high-res PDF)
    out = Path(output_dir) / "kl_control_strategies.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    
    out_pdf = Path(output_dir) / "kl_control_strategies.pdf"
    fig.savefig(out_pdf, dpi=300)
    
    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths_1", type=str, required=True, 
                       help="Comma-separated list of run paths for first plot")
    parser.add_argument("--run_paths_2", type=str, required=True,
                       help="Comma-separated list of run paths for second plot")
    parser.add_argument("--output_dir", type=str, default="plots/kl_control_strategies")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels_1", type=str, default="",
                       help="Comma-separated labels for first plot runs")
    parser.add_argument("--labels_2", type=str, default="",
                       help="Comma-separated labels for second plot runs")
    parser.add_argument("--title_1", type=str, default="KL Control Strategy 1",
                       help="Title for first plot")
    parser.add_argument("--title_2", type=str, default="KL Control Strategy 2",
                       help="Title for second plot")
    args = parser.parse_args()

    run_paths_1 = [p.strip() for p in args.run_paths_1.split(",") if p.strip()]
    run_paths_2 = [p.strip() for p in args.run_paths_2.split(",") if p.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map_1 = build_label_map(run_paths_1, args.labels_1)
    label_map_2 = build_label_map(run_paths_2, args.labels_2)

    plot_two_kl_control(run_paths_1, run_paths_2, args.output_dir, 
                       ema_alpha, args.ci_type, label_map_1, label_map_2,
                       args.title_1, args.title_2)

if __name__ == "__main__":
    main()
