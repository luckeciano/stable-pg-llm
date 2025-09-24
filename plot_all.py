#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List

from plot_utils import DEFAULT_X_METRIC, build_label_map
from plot_train import plot_train_metric
from plot_avg_group import plot_avg_group
from plot_test_grid import plot_test_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_paths", type=str, required=True,
                        help="Comma-separated list of run path folders (each contains subfolders with metrics.csv).")
    parser.add_argument("--x_metric", type=str, default=DEFAULT_X_METRIC)
    parser.add_argument("--avg_group", type=str, default="amc23,math_500,gsm8k")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--ema_alpha", type=float, default=0.25)
    parser.add_argument("--ci_type", type=str, choices=["sem", "bootstrap"], default="sem")
    parser.add_argument("--labels", type=str, default="",
                        help="Comma-separated labels matching the order of --run_paths. If omitted, folder names are used.")
    args = parser.parse_args()

    run_paths: List[str] = [p.strip() for p in args.run_paths.split(",") if p.strip()]
    avg_group = [t.strip() for t in args.avg_group.split(",") if t.strip()]
    ema_alpha = None if args.ema_alpha is None or args.ema_alpha <= 0 else args.ema_alpha
    label_map = build_label_map(run_paths, args.labels)

    # Train
    plot_train_metric(run_paths, args.x_metric, args.output_dir, ema_alpha, args.ci_type, label_map)
    # Avg group
    plot_avg_group(run_paths, args.x_metric, avg_group, args.output_dir, ema_alpha, args.ci_type, label_map)
    # Grid
    plot_test_grid(run_paths, args.x_metric, args.output_dir, ema_alpha, args.ci_type, label_map)

if __name__ == "__main__":
    main()
