#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    DEFAULT_X_METRIC, TRAIN_METRIC, compute_stats, collect_metric_data,
    configure_scientific_x, plot_mean_with_shade, build_label_map, logger, add_start_point
)

REFERENCE_RUN_PATH = "plot_data/Qwen-2.5-7B-Simple-RL-v2"
REFERENCE_X = 60000

def get_reference_value(x_metric: str, ema_alpha: float | None, ci_type: str):
    df = collect_metric_data(REFERENCE_RUN_PATH, x_metric, TRAIN_METRIC)
    df = add_start_point(df, x_metric, y_metric="train/accuracy_reward", mode="avg")
    if df is None:
        logger.warning(f"[reference] Could not load data from {REFERENCE_RUN_PATH}")
        return None

    st = compute_stats(df, x_metric, ema_alpha, ci_type)
    xs, ys = st[x_metric], st["mean"]

    # Find closest point to REFERENCE_X
    idx = np.argmin(np.abs(xs - REFERENCE_X))
    ref_y = ys[idx]
    logger.info(f"[reference] Value at {REFERENCE_X} completions = {ref_y:.4f}")
    return ref_y

def plot_train_metric(run_paths: List[str], x_metric: str, output_dir: str,
                      ema_alpha: float | None, ci_type: str, label_map: Dict[str, str]):
    stats_by_label = {}
    for rp in run_paths:
        df = collect_metric_data(rp, x_metric, TRAIN_METRIC)
        df = add_start_point(df, x_metric, y_metric="train/accuracy_reward", mode="avg")
        if df is None:
            logger.warning(f"[train] Metric not found or could not align in {rp}: {TRAIN_METRIC}")
            continue
        stats_by_label[label_map.get(rp)] = compute_stats(df, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for train/accuracy_reward across all run_paths. Skipping train plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # Add reference line
    ax.set_xlim(0, 11000); ax.set_ylim(0.6, 0.825)
    blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ref_y = get_reference_value(x_metric, ema_alpha, ci_type)
    if ref_y is not None:
        ax.axhline(ref_y, linestyle="--", color=blue, label=None)

        # Add annotation text (right side, just below the line)
        ax.text(
            ax.get_xlim()[1] - 500, ref_y - 0.005,   # right edge in x coords
            r"GRPO at $x = 6 \times 10^4$",
            color=blue,
            fontsize=11,
            va="top", ha="right"   # anchor text to the top-right
        )

    ax.set_xlabel("Training Completions")
    # ax.set_ylabel("Accuracy Reward")
    ax.set_title("MATH Accuracy")
    
    ax.legend(bbox_to_anchor=(0.5, 0.9), loc="lower center", ncol=3, fontsize=12)
    ax.grid(True)
    ax.set_xticks(np.arange(0, 12000, 2000))
    ax.set_yticks(np.arange(0.6, 0.825, 0.05))
    configure_scientific_x(ax)

    fig.tight_layout()
    out = Path(output_dir) / "train_accuracy_reward.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)  # PNG

    # Also save as high-resolution PDF (vector graphics)
    out_pdf = Path(output_dir) / "train_accuracy_reward.pdf"
    fig.savefig(out_pdf, dpi=300)  # dpi optional for PDF, but good for rasterized parts

    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths", type=str, required=True)
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels", type=str, default="")
    args = parser.parse_args()

    run_paths = [p.strip() for p in args.run_paths.split(",") if p.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map = build_label_map(run_paths, args.labels)

    plot_train_metric(run_paths, args.x_metric, args.output_dir, ema_alpha, args.ci_type, label_map)

if __name__ == "__main__":
    main()
