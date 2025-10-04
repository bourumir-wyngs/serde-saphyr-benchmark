#!/usr/bin/env python3
"""
Criterion -> publication-ready plots/tables
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# --------------------------- Discovery ------------------------------------- #

def _discover_groups(root_path: Path) -> List[str]:
    """Find available Criterion 'group' names under criterion/<group>/..."""
    groups = set()
    if root_path.is_file() and root_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(root_path) as z:
            for n in z.namelist():
                if not (n.startswith("criterion/") and n.count("/") >= 2):
                    continue
                parts = n.strip("/").split("/")
                if len(parts) >= 2 and parts[0] == "criterion":
                    groups.add(parts[1])
    else:
        for dirpath, dirs, files in os.walk(root_path):
            if "criterion" in Path(dirpath).parts:
                parts = Path(dirpath).parts
                try:
                    idx = parts.index("criterion")
                except ValueError:
                    continue
                if idx + 1 < len(parts):
                    groups.add(parts[idx+1])
    return sorted(groups)


def _discover_sizes(root_path: Path, group_filter: Optional[str]) -> List[str]:
    """Infer available dataset sizes by scanning for 'sample.json'."""
    sizes=set()
    if root_path.is_file() and root_path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(root_path) as z:
            for n in z.namelist():
                if not (n.endswith("sample.json") and n.startswith("criterion/")):
                    continue
                parts = n.strip("/").split("/")
                if len(parts) < 6:
                    continue
                _, group, _lib, size, _nb, _file = parts[:6]
                if group_filter and group != group_filter:
                    continue
                sizes.add(size)
    else:
        for dirpath, _dirs, files in os.walk(root_path):
            if "sample.json" in files and "criterion" in Path(dirpath).parts:
                parts = Path(dirpath).parts
                try:
                    idx = parts.index("criterion")
                except ValueError:
                    continue
                if idx + 4 >= len(parts):
                    continue
                group, _lib, size, _nb = parts[idx+1:idx+5]
                if group_filter and group != group_filter:
                    continue
                sizes.add(size)
    def size_key(s: str) -> float:
        m = re.findall(r"[0-9.]+", s)
        return float(m[0]) if m else math.inf
    return sorted(sizes, key=size_key)


def _find_sample_files(root_path: Path, group_filter: Optional[str], size_filter: str) -> List[Tuple[str,str,str,io.TextIOBase]]:
    """
    Locate the preferred 'sample.json' per (group, library, size).
    Prefers 'new/sample.json' when both 'new' and 'base' exist.
    """
    results: List[Tuple[str,str,str,io.TextIOBase]] = []
    if root_path.is_file() and root_path.suffix == ".zip":
        import zipfile
        z = zipfile.ZipFile(root_path)
        candidates = [n for n in z.namelist() if n.endswith("sample.json") and n.startswith("criterion/")]
        chosen: Dict[Tuple[str,str,str], Tuple[str,str]] = {}
        for n in candidates:
            parts = n.strip("/").split("/")
            if len(parts) < 6:
                continue
            _, group, lib, size, nb, _file = parts[:6]
            if group_filter and group != group_filter:
                continue
            if size != size_filter:
                continue
            key = (group, lib, size)
            if key not in chosen or (chosen[key][0] != "new" and nb == "new"):
                chosen[key] = (nb, n)
        for (group, lib, size), (_nb, name) in chosen.items():
            fobj = io.TextIOWrapper(io.BytesIO(z.read(name)), encoding="utf-8")
            results.append((group, lib, size, fobj))
        return results
    else:
        pool: Dict[Tuple[str,str,str], Path] = {}
        for dirpath, _dirs, files in os.walk(root_path):
            if "sample.json" not in files:
                continue
            parts = Path(dirpath).parts
            try:
                idx = parts.index("criterion")
            except ValueError:
                continue
            if idx + 4 >= len(parts):
                continue
            group, lib, size, nb = parts[idx+1: idx+5]
            if group_filter and group != group_filter:
                continue
            if size != size_filter:
                continue
            key = (group, lib, size)
            if key not in pool or (pool[key].parts[-2] != "new" and nb == "new"):
                pool[key] = Path(dirpath) / "sample.json"
        for (group, lib, size), path in pool.items():
            results.append((group, lib, size, open(path, "r", encoding="utf-8")))
        return results


def _find_estimate_files(root_path: Path, group_filter: Optional[str], size_filter: str) -> List[Tuple[str,str,str,io.TextIOBase]]:
    """Locate 'estimates.json' per (group, library, size), preferring 'new/'."""
    results: List[Tuple[str,str,str,io.TextIOBase]] = []
    if root_path.is_file() and root_path.suffix == ".zip":
        import zipfile
        z = zipfile.ZipFile(root_path)
        candidates = [n for n in z.namelist() if n.endswith("estimates.json") and n.startswith("criterion/")]
        chosen: Dict[Tuple[str,str,str], Tuple[str,str]] = {}
        for n in candidates:
            parts = n.strip("/").split("/")
            if len(parts) < 6:
                continue
            _, group, lib, size, nb, _file = parts[:6]
            if group_filter and group != group_filter:
                continue
            if size != size_filter:
                continue
            key = (group, lib, size)
            if key not in chosen or (chosen[key][0] != "new" and nb == "new"):
                chosen[key] = (nb, n)
        for (group, lib, size), (_nb, name) in chosen.items():
            fobj = io.TextIOWrapper(io.BytesIO(z.read(name)), encoding="utf-8")
            results.append((group, lib, size, fobj))
        return results
    else:
        pool: Dict[Tuple[str,str,str], Path] = {}
        for dirpath, _dirs, files in os.walk(root_path):
            if "estimates.json" not in files:
                continue
            parts = Path(dirpath).parts
            try:
                idx = parts.index("criterion")
            except ValueError:
                continue
            if idx + 4 >= len(parts):
                continue
            group, lib, size, nb = parts[idx+1: idx+5]
            if group_filter and group != group_filter:
                continue
            if size != size_filter:
                continue
            key = (group, lib, size)
            if key not in pool or (pool[key].parts[-2] != "new" and nb == "new"):
                pool[key] = Path(dirpath) / "estimates.json"
        for (group, lib, size), path in pool.items():
            results.append((group, lib, size, open(path, "r", encoding="utf-8")))
        return results


# --------------------------- Loading --------------------------------------- #

def _pretty_label(lib: str) -> str:
    """
    Normalize library name for nicer plot labels.

    Examples:
      'serde_saphyr_budget_none' -> 'serde-saphyr (budget=none)'
    """
    m = re.match(r"(serde[_-]?saphyr)[_-]?budget[_-]?(.*)", lib)
    if m:
        base = m.group(1).replace("_","-")
        suf  = m.group(2).replace("_","-")
        return f"{base} (budget={suf})"
    x = lib.replace("_","-")
    x = x.replace("serde-yaml-bw", "serde-yaml (bw)")
    x = x.replace("serde-yaml-norway", "serde-yaml (norway)")
    return x


def load_samples(root_path: str, group: Optional[str], sizes: Optional[Sequence[str]]) -> pd.DataFrame:
    """
    Load per-iteration times from Criterion 'sample.json'.

    Returns a DataFrame with:
      ['group','library','label','size','sample_index','time_ms_per_iter'].
    """
    rp = Path(root_path)
    if group is None:
        groups = _discover_groups(rp)
        if len(groups) == 1:
            group = groups[0]
        else:
            raise SystemExit(f"Multiple groups found {groups}. Please pass --group.")
    if sizes is None:
        sizes = _discover_sizes(rp, group)

    records: List[Dict[str,object]] = []
    for size in sizes:
        for g, lib, sz, fobj in _find_sample_files(rp, group, size):
            try:
                data = json.load(fobj)
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass
            times = data.get("times", [])
            iters = data.get("iters", [])
            for i, (t, it) in enumerate(zip(times, iters)):
                per_iter_ms = float(t) / float(it) / 1e6
                records.append({
                    "group": g,
                    "library": lib,
                    "label": _pretty_label(lib),
                    "size": sz,
                    "sample_index": i,
                    "time_ms_per_iter": per_iter_ms,
                })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("No Criterion samples found. Check --group/--sizes and the input path.")
    return df


def load_estimates(root_path: str, group: Optional[str], sizes: Optional[Sequence[str]]) -> pd.DataFrame:
    """
    Load point estimates and confidence intervals from 'estimates.json'.

    Returns a DataFrame with:
      ['group','library','label','size',
       'median_ms','median_ci_low_ms','median_ci_high_ms',
       'mean_ms','mean_ci_low_ms','mean_ci_high_ms']
    """
    rp = Path(root_path)
    if group is None:
        groups = _discover_groups(rp)
        if len(groups) == 1:
            group = groups[0]
        else:
            raise SystemExit(f"Multiple groups found {groups}. Please pass --group.")
    if sizes is None:
        sizes = _discover_sizes(rp, group)

    rows: List[Dict[str,object]] = []
    for size in sizes:
        for g, lib, sz, fobj in _find_estimate_files(rp, group, size):
            try:
                est = json.load(fobj)
            finally:
                try:
                    fobj.close()
                except Exception:
                    pass

            def _extract_ms(key: str) -> Tuple[float,float,float]:
                if key not in est:
                    return float("nan"), float("nan"), float("nan")
                obj = est[key]
                pt = float(obj.get("point_estimate", float("nan"))) / 1e6
                ci = obj.get("confidence_interval", {})
                lo = float(ci.get("lower_bound", float("nan"))) / 1e6
                hi = float(ci.get("upper_bound", float("nan"))) / 1e6
                return pt, lo, hi

            median_ms, median_lo, median_hi = _extract_ms("Median")
            mean_ms, mean_lo, mean_hi = _extract_ms("Mean")

            rows.append({
                "group": g,
                "library": lib,
                "label": _pretty_label(lib),
                "size": sz,
                "median_ms": median_ms,
                "median_ci_low_ms": median_lo,
                "median_ci_high_ms": median_hi,
                "mean_ms": mean_ms,
                "mean_ci_low_ms": mean_lo,
                "mean_ci_high_ms": mean_hi,
            })
    df = pd.DataFrame(rows)
    return df


# --------------------------- Plot helpers ---------------------------------- #

def _size_sort_key(s: str) -> float:
    m = re.findall(r"[0-9.]+", s)
    return float(m[0]) if m else math.inf


def _place_legend_outside(ax: plt.Axes, where: str):
    """Place legend outside the axes (left/right/none)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles or where == "none":
        return None
    if where == "right":
        return ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                         borderaxespad=0.0, frameon=False)
    elif where == "left":
        return ax.legend(handles, labels, loc="center right", bbox_to_anchor=(-0.02, 0.5),
                         borderaxespad=0.0, frameon=False)
    else:
        return ax.legend(handles, labels)


def _compute_point_and_ci_from_samples(sub: pd.DataFrame, ci_level: float, how: str) -> Tuple[float,float,float]:
    """
    From per-iteration samples, compute point + interval.

    Parameters
    ----------
    sub : pandas.DataFrame
        Must have column 'time_ms_per_iter'.
    ci_level : float
        Confidence level in percent (e.g., 95).
    how : str
        'ci' for percentile CI, 'iqr' for [p25, p75], 'none' to return NaNs.

    Returns
    -------
    Tuple[float,float,float]
        (point, low, high) where point is median of samples in ms.
    """
    vals = sub["time_ms_per_iter"].to_numpy()
    if how == "none" or len(vals) == 0:
        m = float(np.median(vals)) if len(vals) else float("nan")
        return m, float("nan"), float("nan")
    if how == "iqr":
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        return float(q50), float(q25), float(q75)
    alpha = 100 - ci_level
    lo, med, hi = np.percentile(vals, [alpha/2.0, 50, 100 - alpha/2.0])
    return float(med), float(lo), float(hi)


def _samples_by_label_size(df: pd.DataFrame) -> Dict[Tuple[str,str], np.ndarray]:
    """Build an index (label,size) -> times array (ms)."""
    out: Dict[Tuple[str,str], np.ndarray] = {}
    for (lab, size), sub in df.groupby(["label","size"]):
        out[(lab, size)] = sub["time_ms_per_iter"].to_numpy()
    return out


# --------------------------- Plotting -------------------------------------- #

def save_boxplots(
        df: pd.DataFrame,
        out_dir: str,
        y_log: bool=False,
        dpi: int=150,
        rotate_labels_90: bool=True
) -> List[str]:
    """Save one box-and-whisker plot per dataset size (labels rotated 90°)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for size, sub in df.groupby("size"):
        meds = sub.groupby("label")["time_ms_per_iter"].median().sort_values()
        order = meds.index.tolist()
        data = [sub.loc[sub["label"] == lab, "time_ms_per_iter"].values for lab in order]

        fig_w = max(6.0, 0.35*len(order) + 3.0)
        fig = plt.figure(figsize=(fig_w, 5.0))
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=order, showfliers=True)
        ax.set_title(f"Group: {sub['group'].iloc[0]} — size: {size}")
        ax.set_ylabel("Time per iteration [ms]")
        ax.set_xlabel("Library")
        if y_log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        if rotate_labels_90:
            ax.set_xticklabels(order, rotation=90)
            ax.tick_params(axis="x", labelsize=8)

        path = Path(out_dir) / f"boxplot_{sub['group'].iloc[0]}_{size}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_median_vs_size_lineplot(
        df: pd.DataFrame,
        out_path: str,
        dpi: int=150,
        legend_outside: str="right",
        error_style: str="ci",
        ci_level: float=95.0,
        use_estimates: str="none",
        estimates_df: Optional[pd.DataFrame]=None,
        marker_size: float=3.0
) -> str:
    """
    Save a line chart of median time vs dataset size with error bars.

    - If use_estimates=='median' or 'mean', pulls point+CI from estimates.json.
    - Else computes nonparametric percentiles from samples (or IQR).
    """
    sizes = sorted(df["size"].unique().tolist(), key=_size_sort_key)
    labels = sorted(df["label"].unique())

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if use_estimates != "none" and estimates_df is None:
        raise ValueError("use_estimates is set but estimates_df is None.")

    for lab in labels:
        y = []; lo = []; hi = []
        for s in sizes:
            if use_estimates == "none":
                sub = df[(df["label"] == lab) & (df["size"] == s)]
                point, low, high = _compute_point_and_ci_from_samples(sub, ci_level, error_style)
            else:
                row = estimates_df[(estimates_df["label"] == lab) & (estimates_df["size"] == s)]
                if row.empty:
                    point, low, high = float("nan"), float("nan"), float("nan")
                else:
                    if use_estimates == "median":
                        point = float(row["median_ms"].iloc[0])
                        low   = float(row["median_ci_low_ms"].iloc[0])
                        high  = float(row["median_ci_high_ms"].iloc[0])
                    else:
                        point = float(row["mean_ms"].iloc[0])
                        low   = float(row["mean_ci_low_ms"].iloc[0])
                        high  = float(row["mean_ci_high_ms"].iloc[0])
            y.append(point); lo.append(low); hi.append(high)

        if error_style != "none" or use_estimates != "none":
            yerr = np.vstack([np.array(y) - np.array(lo), np.array(hi) - np.array(y)])
            ax.errorbar(
                sizes, y, yerr=yerr,
                marker=".", markersize=marker_size, linestyle="-",
                capsize=4, elinewidth=1, capthick=1,
                label=lab
            )
        else:
            ax.plot(sizes, y, marker=".", markersize=marker_size, label=lab)

    ax.set_title(f"Median time vs size — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Median time per iteration [ms]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _place_legend_outside(ax, legend_outside)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_relative_vs_baseline(
        df: pd.DataFrame,
        out_path: str,
        baseline_label: Optional[str]=None,
        dpi: int=150,
        legend_outside: str="right",
        with_ci: bool=True,
        bootstrap_iters: int=2000,
        random_seed: int=42,
        marker_size: float=3.0
) -> str:
    """
    Save a line chart of relative median time vs a baseline library, with error bars.

    - If with_ci=True (default), draws percentile bootstrap CIs for the ratio of medians.
    """
    rng = np.random.default_rng(random_seed)

    sizes = sorted(df["size"].unique().tolist(), key=_size_sort_key)
    med = df.groupby(["label","size"])["time_ms_per_iter"].median().reset_index()

    # Determine baseline per size
    baselines: Dict[str,str] = {}
    for s in sizes:
        if baseline_label is None:
            sub = med[med["size"] == s]
            row = sub.iloc[sub["time_ms_per_iter"].argmin()]
            baselines[s] = str(row["label"])
        else:
            baselines[s] = baseline_label

    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = sorted(med["label"].unique())
    samples = _samples_by_label_size(df)

    for lab in labels:
        rel = []; rel_lo = []; rel_hi = []
        for s in sizes:
            base_lab = baselines[s]

            # ratio of medians (point)
            bm = float(med[(med["label"] == base_lab) & (med["size"] == s)]["time_ms_per_iter"].iloc[0])
            lm = float(med[(med["label"] == lab) & (med["size"] == s)]["time_ms_per_iter"].iloc[0])
            ratio = lm / bm
            rel.append(ratio)

            if with_ci:
                x = samples.get((lab, s), np.array([]))
                b = samples.get((base_lab, s), np.array([]))
                if len(x) == 0 or len(b) == 0:
                    rel_lo.append(np.nan); rel_hi.append(np.nan)
                else:
                    # bootstrap ratio of medians
                    nx, nb = len(x), len(b)
                    idx_x = rng.integers(0, nx, size=(bootstrap_iters, nx))
                    idx_b = rng.integers(0, nb, size=(bootstrap_iters, nb))
                    med_x = np.median(x[idx_x], axis=1)
                    med_b = np.median(b[idx_b], axis=1)
                    r = med_x / med_b
                    lo, hi = np.percentile(r, [2.5, 97.5])
                    rel_lo.append(lo); rel_hi.append(hi)
            else:
                rel_lo.append(np.nan); rel_hi.append(np.nan)

        if with_ci:
            y = np.array(rel)
            yerr = np.vstack([y - np.array(rel_lo), np.array(rel_hi) - y])
            ax.errorbar(
                sizes, rel, yerr=yerr,
                marker=".", markersize=marker_size, linestyle="-",
                capsize=4, elinewidth=1, capthick=1,
                label=lab
            )
        else:
            ax.plot(sizes, rel, marker=".", markersize=marker_size, label=lab)

    ax.axhline(1.0, linestyle="--", linewidth=1)
    blabel = "fastest per size" if baseline_label is None else baseline_label
    ax.set_title(f"Relative median time vs baseline ({blabel}) — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Relative median time (lower is better)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _place_legend_outside(ax, legend_outside)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_size_to_bytes(size_label: str) -> float:
    """
    Convert a dataset size label like '10MiB' to bytes.
    If parsing fails, assume value is MiB.
    """
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG]?i?B)\s*", size_label)
    if not m:
        m2 = re.findall(r"[0-9.]+", size_label)
        if m2:
            return float(m2[0]) * (1024**2)
        return float("nan")
    val = float(m.group(1))
    unit = m.group(2)
    unit = unit.replace("iB","").replace("B","")
    scale = {"":1, "K":1024, "M":1024**2, "G":1024**3}.get(unit, 1024**2)
    return val * scale


def save_throughput_vs_size_lineplot(
        df: pd.DataFrame,
        out_path: str,
        dpi: int=150,
        legend_outside: str="right",
        error_style: str="ci",
        ci_level: float=95.0,
        marker_size: float=3.0
) -> str:
    """
    Save a line chart of median throughput (MiB/s) vs dataset size, with error bars.

    Error bars are percentile CIs (or IQR if requested) computed from per-iteration samples.
    """
    sizes = sorted(df["size"].unique().tolist(), key=_size_sort_key)
    labels = sorted(df["label"].unique())
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Per-sample throughput: MiB / seconds
    df2 = df.copy()
    df2["size_bytes"] = df2["size"].map(_parse_size_to_bytes)
    df2["throughput_mib_s"] = (df2["size_bytes"] / (1024**2)) / (df2["time_ms_per_iter"] / 1000.0)

    for lab in labels:
        y = []; lo = []; hi = []
        for s in sizes:
            sub = df2[(df2["label"] == lab) & (df2["size"] == s)]
            vals = sub["throughput_mib_s"].to_numpy()
            if error_style == "none" or len(vals) == 0:
                median = float(np.median(vals)) if len(vals) else float("nan")
                y.append(median); lo.append(np.nan); hi.append(np.nan)
            elif error_style == "iqr":
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                y.append(float(q50)); lo.append(float(q25)); hi.append(float(q75))
            else:
                alpha = 100 - ci_level
                l, m, h = np.percentile(vals, [alpha/2.0, 50, 100 - alpha/2.0])
                y.append(float(m)); lo.append(float(l)); hi.append(float(h))

        if error_style != "none":
            yarr = np.array(y); loarr = np.array(lo); hiarr = np.array(hi)
            yerr = np.vstack([yarr - loarr, hiarr - yarr])
            ax.errorbar(
                sizes, y, yerr=yerr,
                marker=".", markersize=marker_size, linestyle="-",
                capsize=4, elinewidth=1, capthick=1,
                label=lab
            )
        else:
            ax.plot(sizes, y, marker=".", markersize=marker_size, label=lab)

    ax.set_title(f"Median throughput vs size — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Throughput [MiB/s]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _place_legend_outside(ax, legend_outside)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------- Tables ---------------------------------------- #

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-library summary statistics per size.

    Returns a DataFrame with columns:
      ['size','label','n','median_ms','p25_ms','p75_ms','mean_ms','std_ms',
       'min_ms','max_ms','best_median_ms','rel_to_best','speedup_vs_best']
    """
    rows: List[Dict[str,object]] = []
    for size, sub in df.groupby("size"):
        grp = sub.groupby("label")["time_ms_per_iter"]
        med = grp.median()
        p25 = grp.quantile(0.25)
        p75 = grp.quantile(0.75)
        mean = grp.mean()
        std = grp.std(ddof=1)
        mn = grp.min()
        mx = grp.max()
        n = grp.count()
        best = float(med.min())
        for lab in med.index:
            rows.append({
                "size": size,
                "label": lab,
                "n": int(n[lab]),
                "median_ms": float(med[lab]),
                "p25_ms": float(p25[lab]),
                "p75_ms": float(p75[lab]),
                "mean_ms": float(mean[lab]),
                "std_ms": float(std[lab]),
                "min_ms": float(mn[lab]),
                "max_ms": float(mx[lab]),
                "best_median_ms": best,
                "rel_to_best": float(med[lab]/best),
                "speedup_vs_best": float(best/med[lab]),
            })
    out = pd.DataFrame(rows).sort_values(["size","median_ms"], ascending=[True, True])
    return out


def write_tables(summary_df: pd.DataFrame, out_dir: str) -> Tuple[str,str]:
    """Write summary CSV and Markdown files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / "summary.csv"
    md_path  = Path(out_dir) / "summary.md"
    summary_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        for size, sub in summary_df.groupby("size"):
            sub = sub.sort_values("median_ms")
            f.write(f"### Size: {size}\n\n")
            f.write("| Rank | Library | Median [ms] | p25 [ms] | p75 [ms] | Rel. to best |\n")
            f.write("|---:|---|---:|---:|---:|---:|\n")
            for rank, row in enumerate(sub.itertuples(index=False), start=1):
                f.write(
                    f"| {rank} | {row.label} | "
                    f"{row.median_ms:.3f} | {row.p25_ms:.3f} | {row.p75_ms:.3f} | {row.rel_to_best:.3f}x |\n"
                )
            f.write("\n")
    return str(csv_path), str(md_path)


# --------------------------- CLI ------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    CLI entry point.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Command-line arguments. If None, argparse uses sys.argv.

    Returns
    -------
    None
        Writes figures/tables to disk.
    """
    p = argparse.ArgumentParser(description="Make boxplots and summary tables from Criterion output.")
    p.add_argument("path", help="Path to Criterion results directory OR .zip archive.")
    p.add_argument("--group", default=None, help="Criterion group to include (e.g. 'yaml_parse'). If omitted and only one group exists, it is used.")
    p.add_argument("--sizes", nargs="*", default=None, help="Dataset sizes to include (e.g. 1MiB 5MiB 10MiB). Default: all.")
    p.add_argument("--outdir", default="figures", help="Directory to write figures/tables. Default: ./figures")
    p.add_argument("--box-log-y", action="store_true", help="Use logarithmic Y axis for boxplots.")

    # Line-chart styling / stats
    p.add_argument("--legend-outside", choices=["right","left","none"], default="right", help="Place legend outside plot area.")
    p.add_argument("--error-style", choices=["ci","iqr","none"], default="ci", help="Error bars for median/throughput charts: percentile CI (ci), IQR, or none.")
    p.add_argument("--ci-level", type=float, default=95.0, help="Confidence level for percentile CIs (only if --error-style=ci).")
    p.add_argument("--use-estimates", choices=["none","median","mean"], default="none", help="Use estimates.json for point and CI on the median-vs-size chart.")
    p.add_argument("--marker-size", type=float, default=3.0, help="Marker size for line charts when using '.' marker.")

    # Which charts
    p.add_argument("--no-median-line", action="store_true", help="Skip the 'median vs size' line chart.")
    p.add_argument("--baseline", default=None, help="Baseline label for relative chart (use pretty label, e.g. 'serde-yml'). Default: fastest per size.")
    p.add_argument("--no-relative", action="store_true", help="Skip the relative-vs-baseline chart.")

    # Relative CI toggles (default ON)
    p.set_defaults(relative_ci=True)
    p.add_argument("--relative-ci", dest="relative_ci", action="store_true", help="Draw bootstrap CIs on relative-vs-baseline (default).")
    p.add_argument("--no-relative-ci", dest="relative_ci", action="store_false", help="Disable bootstrap CIs on relative-vs-baseline.")
    p.add_argument("--relative-bootstrap", type=int, default=2000, help="Bootstrap iterations for relative CIs.")
    p.add_argument("--relative-seed", type=int, default=42, help="RNG seed for bootstrap CIs.")

    args = p.parse_args(argv)

    # Load data
    df = load_samples(args.path, args.group, args.sizes)

    # estimates.json if requested
    est_df = None
    if args.use_estimates != "none":
        est_df = load_estimates(args.path, args.group, args.sizes)

    # 1) Boxplots (rotate labels 90° by default)
    box_dir = Path(args.outdir) / "boxplots"
    box_paths = save_boxplots(
        df,
        str(box_dir),
        y_log=args.box_log_y,
        rotate_labels_90=True
    )

    # 2) Median vs size (error bars + '.' marker)
    if not args.no_median_line:
        median_line_path = Path(args.outdir) / "median_vs_size.png"
        save_median_vs_size_lineplot(
            df,
            str(median_line_path),
            legend_outside=args.legend_outside,
            error_style=args.error_style,
            ci_level=args.ci_level,
            use_estimates=args.use_estimates,
            estimates_df=est_df,
            marker_size=args.marker_size
        )
    else:
        median_line_path = None

    # 3) Relative vs baseline (bootstrap CIs ON by default + '.' marker)
    if not args.no_relative:
        rel_path = Path(args.outdir) / "relative_vs_baseline.png"
        save_relative_vs_baseline(
            df,
            str(rel_path),
            baseline_label=args.baseline,
            legend_outside=args.legend_outside,
            with_ci=args.relative_ci,
            bootstrap_iters=args.relative_bootstrap,
            random_seed=args.relative_seed,
            marker_size=args.marker_size
        )
    else:
        rel_path = None

    # 3.5) Throughput vs size (with error bars + '.' marker)
    thr_path = Path(args.outdir) / "throughput_vs_size.png"
    save_throughput_vs_size_lineplot(
        df,
        str(thr_path),
        legend_outside=args.legend_outside,
        error_style=args.error_style,
        ci_level=args.ci_level,
        marker_size=args.marker_size
    )

    # 4) Tables
    summary = build_summary_table(df)
    csv_path, md_path = write_tables(summary, args.outdir)

    # Console output summary
    print("Wrote:")
    for pth in box_paths:
        print(" -", pth)
    if median_line_path:
        print(" -", median_line_path)
    if rel_path:
        print(" -", rel_path)
    print(" -", thr_path)
    print(" -", csv_path)
    print(" -", md_path)


if __name__ == "__main__":
    main()
