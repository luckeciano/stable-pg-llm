#!/usr/bin/env python3
"""
Plot mean and median with confidence intervals across multiple W&B runs per run_path.

Each run_path contains subfolders (one per run) each with a metrics.csv file.
Use --ci_type to specify confidence interval type: 'sem', 'se', or 'bootstrap'.
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set Seaborn style (dark background)
sns.set_style('darkgrid')
# Set the color palette to 'colorblind'
sns.set_palette('colorblind')


def exponential_moving_average(data, alpha=0.95):
    """
    Apply exponential moving average smoothing to data.
    
    Args:
        data: pandas Series or numpy array to smooth
        alpha: smoothing factor (0 < alpha <= 1), higher values = less smoothing
    
    Returns:
        pandas Series with smoothed data
    """
    if isinstance(data, pd.Series):
        return data.ewm(alpha=alpha, adjust=False).mean()
    else:
        # Convert to pandas Series for consistency
        series = pd.Series(data)
        return series.ewm(alpha=alpha, adjust=False).mean()


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


def bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=1000):
    """
    Compute bootstrap confidence interval for the mean.
    
    Args:
        data: Array-like data to bootstrap
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan
    
    # Remove NaN values
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper


def compute_statistics(df: pd.DataFrame, x_metric: str, ema_alpha=None):
    """
    Computes mean, median, SEM, IQR-based error, and bootstrap confidence intervals for each x value.
    Optionally applies exponential moving average smoothing.
    Returns a DataFrame with:
      [x_metric, mean, mean_sem, mean_ci_lower, mean_ci_upper, median, median_se]
    
    Args:
        df: DataFrame with x_metric and run data
        x_metric: Name of the x-axis metric
        ema_alpha: If provided, apply EMA smoothing with this alpha value
    """
    x = df[x_metric]
    y_values = df.drop(columns=[x_metric])

    # Apply EMA smoothing if requested
    if ema_alpha is not None:
        # Apply EMA to each run individually, then compute statistics
        y_values_smoothed = y_values.copy()
        for col in y_values.columns:
            y_values_smoothed[col] = exponential_moving_average(y_values[col], alpha=ema_alpha)
        y_values = y_values_smoothed

    mean = y_values.mean(axis=1, skipna=True)
    mean_sem = y_values.sem(axis=1, skipna=True)

    median = y_values.median(axis=1, skipna=True)

    # Approximate median SE with 1.253 * std/sqrt(n) ~ IQR method
    q75 = y_values.quantile(0.75, axis=1)
    q25 = y_values.quantile(0.25, axis=1)
    iqr = q75 - q25
    n = y_values.count(axis=1).clip(lower=1)  # avoid divide by 0
    median_se = 1.57 * iqr / np.sqrt(n)  # IQR-based approx for median SE

    # Compute bootstrap confidence intervals for the mean
    mean_ci_lower = []
    mean_ci_upper = []
    
    for idx in range(len(y_values)):
        row_data = y_values.iloc[idx].dropna().values
        ci_lower, ci_upper = bootstrap_confidence_interval(row_data)
        mean_ci_lower.append(ci_lower)
        mean_ci_upper.append(ci_upper)

    return pd.DataFrame({
        x_metric: x,
        "mean": mean,
        "mean_sem": mean_sem,
        "mean_ci_lower": mean_ci_lower,
        "mean_ci_upper": mean_ci_upper,
        "median": median,
        "median_se": median_se
    })


def plot_with_shade(x, y, ci_lower, ci_upper, label, ax):
    """Plot y with confidence interval shading."""
    ax.plot(x, y, label=label)
    ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)


def plot_stat(df_stats_dict, x_metric, y_metric, stat_type, output_dir, ci_type="sem", ema_alpha=None):
    """Creates a single plot for a given stat_type (mean or median) with specified confidence interval type."""
    fig, ax = plt.subplots()

    for run_path, df_stats in df_stats_dict.items():
        label = Path(run_path).name
        x = df_stats[x_metric]
        y = df_stats[stat_type]
        
        # Determine confidence interval bounds based on ci_type
        if stat_type == "mean":
            if ci_type == "bootstrap":
                ci_lower = df_stats["mean_ci_lower"]
                ci_upper = df_stats["mean_ci_upper"]
            elif ci_type == "sem":
                ci_lower = y - df_stats["mean_sem"]
                ci_upper = y + df_stats["mean_sem"]
            else:  # se (standard error)
                ci_lower = y - df_stats["mean_sem"]
                ci_upper = y + df_stats["mean_sem"]
        else:  # median
            ci_lower = y - df_stats["median_se"]
            ci_upper = y + df_stats["median_se"]
        
        plot_with_shade(x, y, ci_lower, ci_upper, label, ax)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    
    # Set title based on confidence interval type and smoothing
    if stat_type == "mean":
        if ci_type == "bootstrap":
            title_suffix = " (95% Bootstrap CI)"
            filename_suffix = "_bootstrap_ci"
        elif ci_type == "sem":
            title_suffix = " (SEM)"
            filename_suffix = "_sem"
        else:  # se
            title_suffix = " (SE)"
            filename_suffix = "_se"
    else:  # median
        title_suffix = " (SE)"
        filename_suffix = "_se"
    
    # Add smoothing indicator to title and filename
    if ema_alpha is not None:
        title_suffix += f" [EMA α={ema_alpha}]"
        filename_suffix += f"_ema_{ema_alpha}"
    
    ax.set_title(f"{y_metric} vs {x_metric} ({stat_type}){title_suffix}")
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=1)
    ax.grid(True)
    fig.tight_layout()
    
    filename = Path(output_dir) / f"{y_metric}_{stat_type}{filename_suffix}.png"

    # Create folder if it doesn't exist
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filename)
    plt.close(fig)
    print(f"✓ Saved {filename}")


def main(run_paths, x_metric, y_metrics, output_dir, ci_type="sem", ema_alpha=0.8):
    os.makedirs(output_dir, exist_ok=True)

    for y_metric in y_metrics:
        df_stats_dict = {}
        for run_path in run_paths:
            df_raw = collect_metric_data(run_path, x_metric, y_metric)
            df_stats = compute_statistics(df_raw, x_metric, ema_alpha)
            df_stats_dict[run_path] = df_stats

        plot_stat(df_stats_dict, x_metric, y_metric, "mean", output_dir, ci_type="sem", ema_alpha=ema_alpha)
        plot_stat(df_stats_dict, x_metric, y_metric, "mean", output_dir, ci_type="bootstrap", ema_alpha=ema_alpha)
        plot_stat(df_stats_dict, x_metric, y_metric, "median", output_dir, ci_type="se", ema_alpha=ema_alpha)
        plot_stat(df_stats_dict, x_metric, y_metric, "median", output_dir, ci_type="bootstrap", ema_alpha=ema_alpha)


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
