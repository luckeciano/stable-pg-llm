#!/usr/bin/env python3
from __future__ import annotations
import glob, logging, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from typing import Optional
import seaborn as sns

# =========================
# Styling (edit freely)
# =========================
sns.set_style("darkgrid")
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.0)

# =========================
# Constants (shared)
# =========================
DEFAULT_X_METRIC = "train/num_completions_total"  # alias accepted below
X_METRIC_ALIASES = ["train/num_completions_total", "train/num_completions/total"]
GLOBAL_STEP = "train/global_step"

TRAIN_METRIC = "train/accuracy_reward"
TEST_CUSTOM_METRICS = [
    "test/custom|olympiadbench|0/extractive_match",
    "test/custom|minervamath|0/extractive_match",
    "test/custom|math_500|0/extractive_match",
    "test/custom|gsm8k|0/extractive_match",
    "test/custom|gpqa:diamond|0/extractive_match",
    "test/custom|amc23|0/extractive_match",
    "test/custom|aime25|0/extractive_match",
    "test/custom|aime24|0/extractive_match",
]

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("plotter")


# =========================
# Label mapping
# =========================
def build_label_map(run_paths: List[str], labels_csv: str | None) -> Dict[str, str]:
    raw = [s.strip() for s in (labels_csv or "").split(",")] if labels_csv else []
    label_map: Dict[str, str] = {}

    def _uniqueify(lbl: str, used: set[str]) -> str:
        if lbl not in used:
            return lbl
        i = 2
        while f"{lbl} #{i}" in used:
            i += 1
        return f"{lbl} #{i}"

    used: set[str] = set()
    for i, rp in enumerate(run_paths):
        fallback = Path(rp).name
        cand = raw[i] if i < len(raw) and raw[i] else fallback
        cand = _uniqueify(cand, used)
        used.add(cand)
        label_map[rp] = cand

    if raw and len(raw) != len(run_paths):
        logger.warning(f"--labels count ({len(raw)}) != --run_paths count ({len(run_paths)}). "
                       f"Extra labels ignored; missing labels fall back to folder names.")
    return label_map


# =========================
# Helpers: smoothing, alignment, averaging
# =========================
def exponential_moving_average(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def _resolve_col(df: pd.DataFrame, target: str, aliases: List[str] = None) -> str | None:
    candidates = [target] + (aliases or [])
    for c in candidates:
        if c in df.columns:
            return c
    return None


def collect_metric_data(run_path: str, x_metric: str, y_metric: str) -> pd.DataFrame | None:
    """Align (x,y) per seed on GLOBAL_STEP and return wide df: [x_metric, seed1, seed2, ...]."""
    csv_files = glob.glob(os.path.join(run_path, "*", "metrics.csv"))
    series_list = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
            continue

        x_col = _resolve_col(df, x_metric, X_METRIC_ALIASES)
        if x_col is None or GLOBAL_STEP not in df.columns:
            continue

        if y_metric not in df.columns:
            logger.debug(f"{y_metric} not in {csv_file}")
            continue

        x_part = df[[GLOBAL_STEP, x_col]].dropna(subset=[x_col]).rename(columns={x_col: "__x__"})
        y_part = df[[GLOBAL_STEP, y_metric]].dropna(subset=[y_metric])

        run_id = Path(csv_file).parent.name
        y_part = y_part.rename(columns={y_metric: run_id})

        merged = pd.merge(x_part, y_part, on=GLOBAL_STEP, how="inner")
        if merged.empty:
            continue

        merged = merged[["__x__", run_id]].rename(columns={"__x__": x_metric})
        series_list.append(merged)

    if not series_list:
        return None

    merged_all = series_list[0]
    for s in series_list[1:]:
        merged_all = pd.merge(merged_all, s, on=x_metric, how="outer")

    return merged_all.sort_values(x_metric).reset_index(drop=True)


def resolve_avg_metrics_from_tokens(columns: List[str], tokens: List[str], pool: List[str]) -> List[str]:
    chosen = []
    for t in tokens:
        t = t.strip()
        matches = [m for m in pool if t in m]
        matches = [m for m in matches if m in columns]
        for m in matches:
            if m not in chosen:
                chosen.append(m)
    return chosen


def collect_avg_group_data(run_path: str, x_metric: str, tokens: List[str], pool: List[str]) -> Tuple[pd.DataFrame | None, List[str]]:
    """Per-seed mean across matched metrics, aligned on GLOBAL_STEP; return wide df and list of used metrics."""
    csv_files = glob.glob(os.path.join(run_path, "*", "metrics.csv"))
    series_list = []
    used = set()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")
            continue

        x_col = _resolve_col(df, x_metric, X_METRIC_ALIASES)
        if x_col is None or GLOBAL_STEP not in df.columns:
            continue

        cols = list(df.columns)
        matched = resolve_avg_metrics_from_tokens(cols, tokens, pool)
        if not matched:
            continue
        used.update(matched)

        x_part = df[[GLOBAL_STEP, x_col]].dropna(subset=[x_col]).rename(columns={x_col: "__x__"})
        y_part = df[[GLOBAL_STEP] + matched].copy()
        mask_any = y_part[matched].notna().any(axis=1)
        y_part = y_part.loc[mask_any]
        y_part = pd.DataFrame({
            GLOBAL_STEP: y_part[GLOBAL_STEP],
            "__yavg__": y_part[matched].mean(axis=1, skipna=True)
        })

        merged = pd.merge(x_part, y_part, on=GLOBAL_STEP, how="inner")
        if merged.empty:
            continue

        run_id = Path(csv_file).parent.name
        merged = merged[["__x__", "__yavg__"]].rename(columns={"__x__": x_metric, "__yavg__": run_id})
        series_list.append(merged)

    if not series_list:
        return None, []

    merged_all = series_list[0]
    for s in series_list[1:]:
        merged_all = pd.merge(merged_all, s, on=x_metric, how="outer")
    merged_all = merged_all.sort_values(x_metric).reset_index(drop=True)
    return merged_all, sorted(list(used))


# =========================
# Statistical functions
# =========================
def bootstrap_ci_1d(data: np.ndarray, confidence=0.95, n_bootstrap=1000) -> Tuple[float, float]:
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=data.size, replace=True)
        means.append(sample.mean())
    alpha = 1 - confidence
    return (
        np.percentile(means, 100 * (alpha / 2)),
        np.percentile(means, 100 * (1 - alpha / 2)),
    )


def compute_stats(df: pd.DataFrame, x_metric: str, ema_alpha: float | None, ci_type: str):
    """
    Compute mean and CI across runs at each x.
    Returns DataFrame with columns: [x_metric, mean, lo, hi]
    """
    x = df[x_metric]
    Y = df.drop(columns=[x_metric])

    if ema_alpha and ema_alpha > 0:
        Y = Y.apply(lambda col: exponential_moving_average(col, ema_alpha))

    mean = Y.mean(axis=1, skipna=True)
    sem = Y.sem(axis=1, skipna=True)

    if ci_type == "bootstrap":
        lo, hi = [], []
        for i in range(len(Y)):
            row = Y.iloc[i].dropna().values
            l, h = bootstrap_ci_1d(row)
            lo.append(l); hi.append(h)
        lo = pd.Series(lo, index=mean.index)
        hi = pd.Series(hi, index=mean.index)
    else:
        lo = mean - sem
        hi = mean + sem

    return pd.DataFrame({x_metric: x, "mean": mean, "lo": lo, "hi": hi})


# =========================
# Plotting functions
# =========================
def configure_scientific_x(ax):
    """Force scientific notation like 3 × 10^3 on x-axis."""
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)


def plot_mean_with_shade(ax, x, y, lo, hi, label: str | None = None):
    ax.plot(x, y, label=label)
    ax.fill_between(x, lo, hi, alpha=0.3)



# =========================
# Fixed startpoint
# =========================
FIXED_STARTPOINT_PATH = "plot_data/Qwen-2.5-7B-GRPO-NoBaseline-v2-NoLR"

def _sorted_seed_dirs(base: str) -> List[Path]:
    # Deterministic order: sort by folder name
    dirs = [p for p in Path(base).glob("*") if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name)

def _value_at_step(df: pd.DataFrame, col: str, step: int) -> Optional[float]:
    """Return the last non-NaN value in `col` at GLOBAL_STEP==step, or None if none exists."""
    if GLOBAL_STEP not in df.columns or col not in df.columns:
        return None
    rows = df.loc[(df[GLOBAL_STEP] == step) & df[col].notna(), col]
    if rows.empty:
        return None
    return float(rows.iloc[-1])

def _avg_tokens_at_step(df: pd.DataFrame, tokens: List[str], pool: List[str], step: int) -> Optional[float]:
    if GLOBAL_STEP not in df.columns:
        return None
    matched = resolve_avg_metrics_from_tokens(list(df.columns), tokens, pool)
    if not matched:
        return None
    # only rows at the desired step; average across metrics (skipna)
    rows = df.loc[df[GLOBAL_STEP] == step, matched]
    if rows.empty:
        return None
    vals = rows.iloc[-1].astype(float)
    if vals.notna().any():
        return float(vals.mean(skipna=True))
    return None

def _collect_fixed_seed_values(
    *,
    fixed_path: str,
    step: int,
    y_metric: str | None,
    tokens: List[str] | None,
    pool: List[str] | None,
) -> List[Optional[float]]:
    """Return a list of per-seed values from the fixed path at the given step."""
    vals: List[Optional[float]] = []
    for sd in _sorted_seed_dirs(fixed_path):
        csv_files = list(sd.glob("metrics.csv"))
        if not csv_files:
            vals.append(None); continue
        try:
            df_seed = pd.read_csv(csv_files[0])
        except Exception as e:
            logger.warning(f"[add_start_point] Could not read {csv_files[0]}: {e}")
            vals.append(None); continue

        if y_metric is not None:
            v = _value_at_step(df_seed, y_metric, step)
        else:
            if tokens is None or pool is None:
                logger.warning("[add_start_point] tokens/pool not provided for avg-group start point.")
                v = None
            else:
                v = _avg_tokens_at_step(df_seed, tokens, pool, step)
        vals.append(v)
    return vals

def add_start_point(
    df: pd.DataFrame | None,
    x_metric: str,
    *,
    fixed_path: str = FIXED_STARTPOINT_PATH,
    y_metric: str | None = None,
    tokens: List[str] | None = None,
    pool: List[str] | None = None,
    step: int = 10,
    mode: Literal["per-seed", "avg"] = "per-seed",
) -> pd.DataFrame | None:
    """
    Prepend a synthetic start row at x=0 using values from a fixed path at GLOBAL_STEP==`step`.

    Modes:
      - "per-seed" (default): use the first k fixed seeds to match the k run columns in df.
      - "avg": compute the average across *all* available fixed seeds and assign that same value
               to every run column in df.

    Use with either a single y metric or an avg-group (tokens+pool).
    """
    if df is None:
        return None

    # Identify the run columns in the *target* df and keep a deterministic order
    run_cols = [c for c in df.columns if c != x_metric]
    if not run_cols:
        return df
    run_cols = sorted(run_cols)

    # Collect fixed values per seed
    per_seed_vals = _collect_fixed_seed_values(
        fixed_path=fixed_path, step=step, y_metric=y_metric, tokens=tokens, pool=pool
    )
    if not per_seed_vals:
        logger.warning(f"[add_start_point] No seed dirs found under fixed_path: {fixed_path}")
        return df

    start_row = {x_metric: 0.0}

    if mode == "avg":
        # Average across all available fixed seeds, ignoring NaNs
        arr = pd.Series(per_seed_vals, dtype="float64")
        avg_val = float(arr.mean(skipna=True)) if arr.notna().any() else np.nan
        for col in run_cols:
            start_row[col] = avg_val
    else:
        # "per-seed": align first k seeds to first k run columns
        k = len(run_cols)
        vals = per_seed_vals[:k]
        for i, col in enumerate(run_cols):
            start_row[col] = vals[i] if i < len(vals) else np.nan

    # Concat and sort by x
    start_df = pd.DataFrame([start_row], columns=[x_metric] + run_cols)
    out = pd.concat([start_df, df], ignore_index=True).sort_values(x_metric).reset_index(drop=True)
    return out