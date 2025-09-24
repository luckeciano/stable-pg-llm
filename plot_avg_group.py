#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    DEFAULT_X_METRIC, TEST_CUSTOM_METRICS,
    collect_avg_group_data, compute_stats, configure_scientific_x, plot_mean_with_shade,
    build_label_map, logger, add_start_point
)

REFERENCE_RUN_PATH = "plot_data/Qwen-2.5-7B-Simple-RL-v2"
REFERENCE_X = 60000

def get_reference_value_avg(x_metric: str, avg_group: List[str],
                            ema_alpha: float | None, ci_type: str):
    """
    Compute the reference y-value at REFERENCE_X for the averaged group metric
    from the fixed REFERENCE_RUN_PATH.
    """
    df_avg, used = collect_avg_group_data(REFERENCE_RUN_PATH, x_metric, avg_group, TEST_CUSTOM_METRICS)
    df_avg = add_start_point(df_avg, x_metric, tokens=avg_group, pool=TEST_CUSTOM_METRICS, mode="avg")
    if df_avg is None:
        logger.warning(f"[reference-avg] No matched/aligned data in {REFERENCE_RUN_PATH} for tokens: {avg_group}")
        return None

    st = compute_stats(df_avg, x_metric, ema_alpha, ci_type)
    xs, ys = st[x_metric], st["mean"]

    idx = np.argmin(np.abs(xs - REFERENCE_X))
    ref_y = ys[idx]
    logger.info(f"[reference-avg] Value at {REFERENCE_X} completions = {ref_y:.4f}")
    return ref_y

def plot_avg_group(run_paths: List[str], x_metric: str, avg_group: List[str], output_dir: str,
                   ema_alpha: float | None, ci_type: str, label_map: Dict[str, str]):
    if not avg_group:
        logger.info("No --avg_group provided. Skipping avg_group figure.")
        return

    stats_by_label = {}
    used_any: set[str] = set()

    for rp in run_paths:
        df_avg, used = collect_avg_group_data(rp, x_metric, avg_group, TEST_CUSTOM_METRICS)
        df_avg = add_start_point(df_avg, x_metric, tokens=avg_group, pool=TEST_CUSTOM_METRICS, mode="avg")
        if df_avg is None:
            logger.warning(f"[avg_group] No matched/aligned data in {rp} for tokens: {avg_group}")
            continue
        used_any.update(used)
        stats_by_label[label_map.get(rp)] = compute_stats(df_avg, x_metric, ema_alpha, ci_type)

    if not stats_by_label:
        logger.warning("No data found for avg_group across all run_paths. Skipping avg_group plot.")
        return

    # ----- Plot lines -----
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, st in stats_by_label.items():
        plot_mean_with_shade(ax, st[x_metric], st["mean"], st["lo"], st["hi"], label)

    # Axis labels / title consistent with plot_fig_1.py
    ax.set_xlabel("Training Completions")
    # ax.set_ylabel(human_label)   # keep minimalist style like plot_fig_1.py
    ax.set_title(f"TEST Accuracy")

    # X/Y formatting similar to plot_fig_1.py (keep x-range consistent if desired)
    ax.set_xlim(0, 10500)
    ax.grid(True)
    configure_scientific_x(ax)

    # Legend placement similar to plot_fig_1.py
    ax.legend(bbox_to_anchor=(0.5, 0.9), loc="lower center", ncol=3, fontsize=12)

    ax.set_xlim(0, 10500); 
    ax.set_ylim(0.32, 0.42)

    # ----- Reference dashed line + annotation in first palette color -----
    blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ref_y = get_reference_value_avg(x_metric, avg_group, ema_alpha, ci_type)
    if ref_y is not None:
        ax.axhline(ref_y, linestyle="--", color=blue, label=None)

        # Place annotation on the right, slightly inset and just below the line
        xmax, xmin = ax.get_xlim()[1], ax.get_xlim()[0]
        xoffset = 0.05 * (xmax - xmin)  # 5% inward from the right
        ax.text(
            xmax - xoffset, ref_y - 0.005,
            r"GRPO at $x = 6 \times 10^4$",
            color=blue,
            fontsize=11,
            va="top", ha="right"
        )

    # ----- Save outputs (PNG + high-res PDF) -----
    fig.tight_layout()
    out = Path(output_dir) / f"avg_{'_'.join(avg_group)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)

    out_pdf = Path(output_dir) / f"avg_{'_'.join(avg_group)}.pdf"
    fig.savefig(out_pdf, dpi=300)  # dpi optional for vector PDF, helpful if any rasterized elements

    if used_any:
        logger.info("avg_group used metrics:")
        for m in sorted(used_any):
            logger.info(f"  - {m}")
    logger.info(f"Saved {out}")
    logger.info(f"Saved {out_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths", type=str, required=True)
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--avg_group", type=str, default="amc23,math_500,gsm8k")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--ema_alpha", type=float, default=0.35)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels", type=str, default="")
    args = parser.parse_args()

    run_paths = [p.strip() for p in args.run_paths.split(",") if p.strip()]
    avg_group = [t.strip() for t in args.avg_group.split(",") if t.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map = build_label_map(run_paths, args.labels)

    plot_avg_group(run_paths, args.x_metric, avg_group, args.output_dir, ema_alpha, args.ci_type, label_map)

if __name__ == "__main__":
    main()
