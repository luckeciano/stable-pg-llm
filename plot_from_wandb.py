#!/usr/bin/env python3
"""
Plot mean and median with standard error across multiple W&B runs per run_path.

Each run_path contains subfolders (one per run) each with a metrics.csv file.
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set Seaborn style (dark background)
sns.set_style('darkgrid')
# Set the color palette to 'colorblind'
sns.set_palette('colorblind')


def collect_metric_data(run_path: str, x_metric: str, y_metric: str) -> pd.DataFrame:
    """
    Loads all metrics.csv under run_path, aligns them by x_metric.
    Returns a DataFrame with columns: [x_metric, run1, run2, ..., runN]
    """
    csv_files = glob.glob(os.path.join(run_path, "*", "metrics.csv"))
    series_list = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if x_metric not in df.columns or y_metric not in df.columns:
            continue
        run_id = Path(csv_file).parent.name
        series = df[[x_metric, y_metric]].dropna()
        series = series.rename(columns={y_metric: run_id})
        series_list.append(series)

    if not series_list:
        raise ValueError(f"No valid runs found in {run_path} with metric {y_metric}.")

    merged_df = series_list[0]
    for series in series_list[1:]:
        merged_df = pd.merge(merged_df, series, on=x_metric, how='outer')

    merged_df = merged_df.sort_values(x_metric).reset_index(drop=True)
    return merged_df


def compute_statistics(df: pd.DataFrame, x_metric: str):
    """
    Computes mean, median, SEM and IQR-based error for each x value.
    Returns a DataFrame with:
      [x_metric, mean, mean_sem, median, median_se]
    """
    x = df[x_metric]
    y_values = df.drop(columns=[x_metric])

    mean = y_values.mean(axis=1, skipna=True)
    mean_sem = y_values.sem(axis=1, skipna=True)

    median = y_values.median(axis=1, skipna=True)

    # Approximate median SE with 1.253 * std/sqrt(n) ~ IQR method
    q75 = y_values.quantile(0.75, axis=1)
    q25 = y_values.quantile(0.25, axis=1)
    iqr = q75 - q25
    n = y_values.count(axis=1).clip(lower=1)  # avoid divide by 0
    median_se = 1.57 * iqr / np.sqrt(n)  # IQR-based approx for median SE

    return pd.DataFrame({
        x_metric: x,
        "mean": mean,
        "mean_sem": mean_sem,
        "median": median,
        "median_se": median_se
    })


def plot_with_shade(x, y, err, label, ax):
    """Plot y ± err with shading."""
    ax.plot(x, y, label=label)
    ax.fill_between(x, y - err, y + err, alpha=0.3)


def plot_stat(df_stats_dict, x_metric, y_metric, stat_type, output_dir):
    """Creates a single plot for a given stat_type (mean or median)."""
    fig, ax = plt.subplots()

    for run_path, df_stats in df_stats_dict.items():
        label = Path(run_path).name
        x = df_stats[x_metric]
        y = df_stats[stat_type]
        err = df_stats[f"{stat_type}_sem"] if stat_type == "mean" else df_stats[f"{stat_type}_se"]
        plot_with_shade(x, y, err, label, ax)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{y_metric} vs {x_metric} ({stat_type})")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    filename = Path(output_dir) / f"{y_metric}_{stat_type}.png"

    # Create folder if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filename)
    plt.close(fig)
    print(f"✓ Saved {filename}")


def main(run_paths, x_metric, y_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for y_metric in y_metrics:
        df_stats_dict = {}
        for run_path in run_paths:
            df_raw = collect_metric_data(run_path, x_metric, y_metric)
            df_stats = compute_statistics(df_raw, x_metric)
            df_stats_dict[run_path] = df_stats

        plot_stat(df_stats_dict, x_metric, y_metric, "mean", output_dir)
        plot_stat(df_stats_dict, x_metric, y_metric, "median", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_paths",
        type=str,
        required=True,
        help="Comma-separated list of run path folders.",
    )
    parser.add_argument(
        "--x_metric",
        type=str,
        required=True,
        help="Metric name to use on the x-axis.",
    )
    parser.add_argument(
        "--y_metrics",
        type=str,
        required=True,
        help="Comma-separated list of y-axis metrics to plot.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Where to save the output figures.",
    )
    args = parser.parse_args()

    run_paths = [p.strip() for p in args.run_paths.split(",")]
    y_metrics = [m.strip() for m in args.y_metrics.split(",")]

    main(run_paths, args.x_metric, y_metrics, args.output_dir)
