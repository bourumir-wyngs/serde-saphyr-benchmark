#!/usr/bin/env python3
"""
Criterion -> publication-ready plots/tables

This script reads Criterion benchmark results (either from a .zip or a directory)
and produces:
  1. Box-and-whisker plots of per-iteration times, one figure per dataset size.
  2. A line chart of median time vs dataset size per library.
  3. An optional line chart of relative time vs a chosen baseline library.
  4. CSV + Markdown tables with summary statistics and relative performance.

It expects Criterion's standard layout:
  criterion/<group>/<library>/<size>/(new|base)/sample.json

Author: <you>
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


# --------------------------- Data Loading ---------------------------------- #

def _discover_sizes(root_path: Path, group_filter: Optional[str]) -> List[str]:
    """
    Infer available dataset sizes by scanning for 'sample.json'.

    Parameters
    ----------
    root_path : Path
        Path to a Criterion results directory or a .zip file.
    group_filter : Optional[str]
        If provided, only sizes under this Criterion group are considered.

    Returns
    -------
    List[str]
        Sorted list of unique size labels (e.g. ['1MiB','5MiB','10MiB']).
    """
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

    Parameters
    ----------
    root_path : Path
        Path to a Criterion results directory or a .zip file.
    group_filter : Optional[str]
        If provided, only consider this Criterion group.
    size_filter : str
        Only include entries matching this dataset size label.

    Returns
    -------
    List[Tuple[str,str,str,io.TextIOBase]]
        List of (group, library, size, opened_text_file) tuples. Call .close()
        on each file-like when done reading.
    """
    results: List[Tuple[str,str,str,io.TextIOBase]] = []
    if root_path.is_file() and root_path.suffix == ".zip":
        import zipfile
        z = zipfile.ZipFile(root_path)
        candidates = [n for n in z.namelist() if n.endswith("sample.json") and n.startswith("criterion/")]
        # pick new over base
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
            # wrap bytes in a TextIO so json.load works
            fobj = io.TextIOWrapper(io.BytesIO(z.read(name)), encoding="utf-8")
            results.append((group, lib, size, fobj))
        return results
    else:
        # Directory walk
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


def _pretty_label(lib: str) -> str:
    """
    Normalize library name for nicer plot labels.

    Parameters
    ----------
    lib : str
        Raw library identifier from Criterion path.

    Returns
    -------
    str
        Human-friendly label (e.g. 'serde-saphyr (budget=none)').
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

    Parameters
    ----------
    root_path : str
        Path to a Criterion results directory or .zip archive.
    group : Optional[str]
        Criterion benchmark group to include (e.g. 'yaml_parse'). If None and
        only one group exists in the input, that group is used.
    sizes : Optional[Sequence[str]]
        Iterable of dataset-size labels to include (e.g. ['10MiB']). If None,
        all available sizes are included.

    Returns
    -------
    pandas.DataFrame
        Columns: ['group','library','label','size','sample_index','time_ms_per_iter'].
    """
    rp = Path(root_path)
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


# --------------------------- Plotting -------------------------------------- #

def save_boxplots(df: pd.DataFrame, out_dir: str, y_log: bool=False, dpi: int=150) -> List[str]:
    """
    Save one box-and-whisker plot per dataset size.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by load_samples().
    out_dir : str
        Directory to write image files into.
    y_log : bool
        If True, use log10 scale on the Y axis.
    dpi : int
        Dots-per-inch for the saved PNGs.

    Returns
    -------
    List[str]
        List of file paths to the saved boxplot PNGs.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for size, sub in df.groupby("size"):
        meds = sub.groupby("label")["time_ms_per_iter"].median().sort_values()
        order = meds.index.tolist()
        data = [sub.loc[sub["label"] == lab, "time_ms_per_iter"].values for lab in order]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=order, showfliers=True)
        ax.set_title(f"Group: {sub['group'].iloc[0]} — size: {size}")
        ax.set_ylabel("Time per iteration [ms]")
        ax.set_xlabel("Library")
        if y_log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.autofmt_xdate(rotation=20)

        path = Path(out_dir) / f"boxplot_{sub['group'].iloc[0]}_{size}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_median_vs_size_lineplot(df: pd.DataFrame, out_path: str, dpi: int=150) -> str:
    """
    Save a line chart of median time vs dataset size, one line per library.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by load_samples().
    out_path : str
        Where to save the PNG file.
    dpi : int
        Dots-per-inch for the saved PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    def size_key(s: str) -> float:
        m = re.findall(r"[0-9.]+", s)
        return float(m[0]) if m else math.inf

    sizes = sorted(df["size"].unique().tolist(), key=size_key)
    med = df.groupby(["label","size"])["time_ms_per_iter"].median().reset_index()
    labels = sorted(med["label"].unique())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lab in labels:
        y = []
        for s in sizes:
            val = med[(med["label"] == lab) & (med["size"] == s)]["time_ms_per_iter"]
            y.append(float(val.iloc[0]) if not val.empty else np.nan)
        ax.plot(sizes, y, marker="o", label=lab)

    ax.set_title(f"Median time vs size — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Median time per iteration [ms]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_relative_vs_baseline(df: pd.DataFrame, out_path: str, baseline_label: Optional[str]=None, dpi: int=150) -> str:
    """
    Save a line chart of relative median time vs a baseline library.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by load_samples().
    out_path : str
        Where to save the PNG file.
    baseline_label : Optional[str]
        If provided, use this label as the baseline across all sizes.
        If None, the fastest (lowest median) per size is used as baseline.
    dpi : int
        Dots-per-inch for the saved PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    def size_key(s: str) -> float:
        m = re.findall(r"[0-9.]+", s)
        return float(m[0]) if m else math.inf
    sizes = sorted(df["size"].unique().tolist(), key=size_key)
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

    labels = sorted(med["label"].unique())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lab in labels:
        rel = []
        for s in sizes:
            base_lab = baselines[s]
            bm = float(med[(med["label"] == base_lab) & (med["size"] == s)]["time_ms_per_iter"].iloc[0])
            lm = float(med[(med["label"] == lab) & (med["size"] == s)]["time_ms_per_iter"].iloc[0])
            rel.append(lm / bm)
        ax.plot(sizes, rel, marker="o", label=lab)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    blabel = "fastest per size" if baseline_label is None else baseline_label
    ax.set_title(f"Relative median time vs baseline ({blabel}) — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Relative median time (lower is better)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ------------------ Throughput (MiB/s) plot -------------------------------- #

def _parse_size_to_bytes(size_label: str) -> float:
    """
    Convert a dataset size label like '10MiB' to bytes.

    Parameters
    ----------
    size_label : str
        Size label as used by the benchmarks (e.g. '10MiB').

    Returns
    -------
    float
        Size in bytes. If parsing fails, returns NaN.
    """
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG]?i?B)\s*", size_label)
    if not m:
        # Try bare number as MiB
        m2 = re.findall(r"[0-9.]+", size_label)
        if m2:
            return float(m2[0]) * (1024**2)
        return float("nan")
    val = float(m.group(1))
    unit = m.group(2)
    unit = unit.replace("iB","").replace("B","")
    scale = {"":1, "K":1024, "M":1024**2, "G":1024**3}.get(unit, 1024**2)
    return val * scale


def save_throughput_vs_size_lineplot(df: pd.DataFrame, out_path: str, dpi: int=150) -> str:
    """
    Save a line chart of median throughput (MiB/s) vs dataset size.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by load_samples().
    out_path : str
        Where to save the PNG file.
    dpi : int
        Dots-per-inch for the saved PNG.

    Returns
    -------
    str
        Path to the saved image.
    """
    def size_key(s: str) -> float:
        m = re.findall(r"[0-9.]+", s)
        return float(m[0]) if m else math.inf
    sizes = sorted(df["size"].unique().tolist(), key=size_key)
    med = df.groupby(["label","size"])["time_ms_per_iter"].median().reset_index()
    # compute throughput in MiB/s
    med["size_bytes"] = med["size"].map(_parse_size_to_bytes)
    med["throughput_mib_s"] = (med["size_bytes"] / (1024**2)) / (med["time_ms_per_iter"] / 1000.0)

    labels = sorted(med["label"].unique())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lab in labels:
        y = []
        for s in sizes:
            row = med[(med["label"] == lab) & (med["size"] == s)]
            y.append(float(row["throughput_mib_s"].iloc[0]) if not row.empty else np.nan)
        ax.plot(sizes, y, marker="o", label=lab)
    ax.set_title(f"Median throughput vs size — group: {df['group'].iloc[0]}")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Throughput [MiB/s]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# --------------------------- Tables ---------------------------------------- #

def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-library summary statistics per size.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by load_samples().

    Returns
    -------
    pandas.DataFrame
        Columns:
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
    """
    Write summary CSV and Markdown files.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output of build_summary_table().
    out_dir : str
        Directory where CSV and Markdown files will be saved.

    Returns
    -------
    Tuple[str,str]
        (csv_path, md_path) to the saved files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / "summary.csv"
    md_path  = Path(out_dir) / "summary.md"
    # CSV with all numeric precision
    summary_df.to_csv(csv_path, index=False)

    # Markdown, prettier columns and sorted per size
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
        This function does not return a value; it writes files to disk.
    """
    p = argparse.ArgumentParser(description="Make boxplots and summary tables from Criterion output.")
    p.add_argument("path", help="Path to Criterion results directory OR .zip archive.")
    p.add_argument("--group", default=None, help="Criterion group to include (e.g. 'yaml_parse'). If omitted and only one group exists, it is used.")
    p.add_argument("--sizes", nargs="*", default=None, help="Dataset sizes to include (e.g. 1MiB 5MiB 10MiB). Default: all.")
    p.add_argument("--outdir", default="figures", help="Directory to write figures/tables. Default: ./figures")
    p.add_argument("--box-log-y", action="store_true", help="Use logarithmic Y axis for boxplots.")
    p.add_argument("--no-median-line", action="store_true", help="Skip the 'median vs size' line chart.")
    p.add_argument("--baseline", default=None, help="Baseline label for relative chart (use pretty label, e.g. 'serde-yml'). Default: fastest per size.")
    p.add_argument("--no-relative", action="store_true", help="Skip the relative-vs-baseline chart.")
    args = p.parse_args(argv)

    df = load_samples(args.path, args.group, args.sizes)

    # 1) Boxplots
    box_dir = Path(args.outdir) / "boxplots"
    box_paths = save_boxplots(df, str(box_dir), y_log=args.box_log_y)

    # 2) Median vs size
    if not args.no_median_line:
        median_line_path = Path(args.outdir) / "median_vs_size.png"
        save_median_vs_size_lineplot(df, str(median_line_path))
    else:
        median_line_path = None

    # 3) Relative vs baseline
    if not args.no_relative:
        rel_path = Path(args.outdir) / "relative_vs_baseline.png"
        save_relative_vs_baseline(df, str(rel_path), baseline_label=args.baseline)
    else:
        rel_path = None

    # 3.5) Throughput vs size
    thr_path = Path(args.outdir) / "throughput_vs_size.png"
    save_throughput_vs_size_lineplot(df, str(thr_path))

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
    thr_path = Path(args.outdir) / "throughput_vs_size.png"
    print(" -", thr_path)
    print(" -", csv_path)
    print(" -", md_path)


if __name__ == "__main__":
    main()
