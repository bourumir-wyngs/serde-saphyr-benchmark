#!/usr/bin/env python3
"""
Summarize Criterion results AND produce combined per-crate plots at a chosen size.

Outputs (in the working directory by default):
  - summary.csv                   (mean/stddev/median & MiB/s across sizes)
  - summary.md                    (pivot: crates × sizes with MiB/s)
  - combined_kde_<SIZE>.png       (Gaussian-smoothed duration densities, all crates)
  - combined_box_<SIZE>.png       (box & whiskers, all crates, fliers hidden)

Assumptions:
  - Script is run from the project root (next to ./target/)
  - Criterion layout:
      ./target/criterion/<group>/<crate>/<size_label>/{new,base}/
        estimates.json
        raw.csv          (preferred for per-sample values)
        sample.json      (fallback for per-sample values)

Usage:
  python3 summary.py
  python3 summary.py --group yaml_parse --size 25MiB --trim 0.25 --q-low 0.10 --q-high 0.99
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# matplotlib only (no seaborn)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- Data types -----------------------------

@dataclass
class Estimate:
    mean_ns: float
    median_ns: float
    stddev_ns: float

@dataclass
class Record:
    group: str
    crate: str
    size_label: str
    size_bytes: int
    est: Estimate


# ----------------------------- Utilities -----------------------------

_SIZE_RE = re.compile(r"^\s*(\d+)\s*MiB\s*$", re.IGNORECASE)

def parse_size_label(size_label: str) -> Optional[int]:
    m = _SIZE_RE.match(size_label)
    if not m:
        return None
    mib = int(m.group(1))
    return mib * 1024 * 1024

def load_estimates_json(json_path: Path) -> Optional[Estimate]:
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        def pe(key: str) -> float:
            return float(data[key]["point_estimate"])

        mean = pe("mean")
        median = pe("median")
        stddev = pe("std_dev")

        # Normalize to ns: Criterion JSON usually stores seconds (small floats) or ns (large).
        def to_ns(x: float) -> float:
            if x > 1e6:
                return x                 # already ns
            if x < 1e3:
                return x * 1e9          # seconds -> ns
            return x                     # ambiguous: treat as ns

        return Estimate(mean_ns=to_ns(mean), median_ns=to_ns(median), stddev_ns=to_ns(stddev))
    except Exception as e:
        print(f"[warn] Failed to parse {json_path}: {e}")
        return None

def human_crate_label(crate_folder: str) -> str:
    # Map friendly labels
    if crate_folder == "serde_saphyr_budget_max":
        return "serde_saphyr (budget=max)"
    if crate_folder == "serde_saphyr_budget_none":
        return "serde_saphyr (budget=none)"
    mapping = {
        "serde_yaml": "serde_yaml",
        "serde_yaml_bw": "serde_yaml_bw",
        "serde_yaml_norway": "serde_yaml_norway",
        "serde_yml": "serde_yml",
        "serde_norway": "serde_norway",
    }
    return mapping.get(crate_folder, crate_folder)

def available_size_dirs(crate_dir: Path) -> List[Path]:
    return [p for p in crate_dir.iterdir() if p.is_dir() and p.name not in {"report"}]


# ------------------------- Core collection ---------------------------

def collect_records(target_dir: Path, group: str) -> List[Record]:
    root = target_dir / "criterion" / group
    if not root.exists():
        raise SystemExit(f"[error] Group directory not found: {root}")

    records: List[Record] = []
    for crate_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "report"):
        for size_dir in sorted(available_size_dirs(crate_dir)):
            size_bytes = parse_size_label(size_dir.name)
            if size_bytes is None:
                continue
            json_new = size_dir / "new" / "estimates.json"
            json_base = size_dir / "base" / "estimates.json"
            est = load_estimates_json(json_new) if json_new.exists() else (
                load_estimates_json(json_base) if json_base.exists() else None
            )
            if est is None:
                continue
            records.append(Record(
                group=group,
                crate=human_crate_label(crate_dir.name),
                size_label=size_dir.name,
                size_bytes=size_bytes,
                est=est,
            ))
    records.sort(key=lambda r: (r.size_bytes, r.crate))
    return records


# --------------------------- Raw samples -----------------------------

def _load_raw_csv(path: Path) -> List[float]:
    vals: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        _ = next(rdr, None)  # header (may or may not exist)
        for row in rdr:
            # take first parseable numeric cell
            for cell in row:
                try:
                    v = float(cell)
                except Exception:
                    continue
                # if tiny (<1e3), assume seconds; else ns
                v_ns = v * 1e9 if v < 1e3 else v
                vals.append(v_ns)
                break
    return vals

def _load_sample_json(path: Path) -> List[float]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    seq = None
    for key in ("values", "samples", "sample", "times", "measured_values"):
        if key in data and isinstance(data[key], list):
            seq = data[key]
            break
    if not isinstance(seq, list):
        return []
    vals: List[float] = []
    for x in seq:
        try:
            v = float(x)
        except Exception:
            continue
        vals.append(v * 1e9 if v < 1e3 else v)
    return vals

def load_samples_for_size(crate_dir: Path, size_label: str) -> Optional[List[float]]:
    """Return list of per-sample durations in ns, or None if not found."""
    size_dir = crate_dir / size_label
    if not size_dir.exists():
        return None
    candidates = [
        size_dir / "new" / "raw.csv",
        size_dir / "base" / "raw.csv",
        size_dir / "new" / "sample.json",
        size_dir / "base" / "sample.json",
        ]
    # Try raw.csv first
    for p in candidates[:2]:
        if p.exists():
            try:
                vals = _load_raw_csv(p)
                if vals:
                    return vals
            except Exception as e:
                print(f"[warn] Failed to parse {p}: {e}")
    # Fallback to sample.json
    for p in candidates[2:]:
        if p.exists():
            try:
                vals = _load_sample_json(p)
                if vals:
                    return vals
            except Exception as e:
                print(f"[warn] Failed to parse {p}: {e}")
    return None


# --------------------------- CSV / Markdown --------------------------

def write_csv(rows: List[Record], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group","crate","size_label","size_bytes","mean_ms","stddev_ms","median_ms","throughput_mib_per_s"])
        for r in rows:
            mean_s = r.est.mean_ns / 1e9
            std_s = r.est.stddev_ns / 1e9
            med_s = r.est.median_ns / 1e9
            mib = r.size_bytes / (1024.0 * 1024.0)
            thr = mib / mean_s if mean_s > 0 else 0.0
            w.writerow([
                r.group, r.crate, r.size_label, r.size_bytes,
                round(mean_s * 1e3, 3),
                round(std_s * 1e3, 3),
                round(med_s * 1e3, 3),
                round(thr, 2),
            ])

def write_markdown_table(rows: List[Record], out_md: Path) -> None:
    sizes_sorted = sorted({(r.size_bytes, r.size_label) for r in rows})
    crates_sorted = sorted({r.crate for r in rows})
    idx: Dict[Tuple[str, str], float] = {}
    for r in rows:
        mean_s = r.est.mean_ns / 1e9
        mib = r.size_bytes / (1024.0 * 1024.0)
        thr = mib / mean_s if mean_s > 0 else 0.0
        idx[(r.crate, r.size_label)] = thr

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as fh:
        fh.write("| Crate | " + " | ".join(lbl for _, lbl in sizes_sorted) + " |\n")
        fh.write("|---|" + "|".join("---" for _ in sizes_sorted) + "|\n")
        for crate in crates_sorted:
            cells = []
            for _, lbl in sizes_sorted:
                thr = idx.get((crate, lbl))
                cells.append(f"{thr:.1f} MiB/s" if thr is not None else "—")
            fh.write(f"| `{crate}` | " + " | ".join(cells) + " |\n")

def print_rankings(rows: List[Record]) -> None:
    by_size: Dict[int, List[Record]] = {}
    for r in rows:
        by_size.setdefault(r.size_bytes, []).append(r)
    print("\nPer-size throughput rankings (higher is better):")
    for size_bytes in sorted(by_size.keys()):
        size_mib = int(size_bytes / (1024 * 1024))
        group = by_size[size_bytes]
        scored = []
        for r in group:
            mean_s = r.est.mean_ns / 1e9
            mib = r.size_bytes / (1024.0 * 1024.0)
            thr = mib / mean_s if mean_s > 0 else 0.0
            scored.append((thr, r.crate))
        scored.sort(reverse=True)
        print(f"\n  {size_mib} MiB:")
        for rank, (thr, crate) in enumerate(scored, start=1):
            print(f"    {rank:>2}. {crate:30s}  {thr:8.2f} MiB/s")


# --------------------------- Plotting utils --------------------------

def gaussian_kde(xs: List[float], grid: List[float]) -> List[float]:
    """
    Simple Gaussian KDE with Silverman's rule bandwidth.
    - xs: samples (ms)
    - grid: x locations (ms) to evaluate
    """
    if not xs:
        return [0.0 for _ in grid]
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) * (x - mean) for x in xs) / max(1, n - 1)
    std = math.sqrt(max(1e-12, var))
    # Silverman's bandwidth
    h = 1.06 * std * (n ** (-1/5)) if std > 0 else 1.0
    inv = 1.0 / (math.sqrt(2.0 * math.pi) * h * n)
    out: List[float] = []
    for x in grid:
        s = 0.0
        for xi in xs:
            u = (x - xi) / h
            s += math.exp(-0.5 * u * u)
        out.append(inv * s)
    return out

def trim_and_clip(samples_ns: List[float], trim_frac: float, q_low: float, q_high: float) -> List[float]:
    """Trim first `trim_frac` portion, then clip to [q_low, q_high] quantiles. Return ms values."""
    if not samples_ns:
        return []
    n = len(samples_ns)
    start = int(n * trim_frac)
    core = samples_ns[start:]

    # Convert to ms
    core_ms = [v / 1e6 for v in core]
    if not core_ms:
        return []

    # Quantile clipping to remove near-zero artefacts and extreme tails
    xs = sorted(core_ms)
    def quantile(arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        q = min(max(q, 0.0), 1.0)
        idx = q * (len(arr) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return arr[lo]
        w = idx - lo
        return arr[lo] * (1 - w) + arr[hi] * w

    lo_v = quantile(xs, q_low)
    hi_v = quantile(xs, q_high)
    clipped = [x for x in core_ms if lo_v <= x <= hi_v]
    return clipped


# --------------------------- Combined plots --------------------------

def plot_combined_kde(target_dir: Path, group: str, size_label: str,
                      trim_frac: float, q_low: float, q_high: float,
                      out_png: Path) -> None:
    root = target_dir / "criterion" / group
    crates = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name != "report"]

    series: Dict[str, List[float]] = {}
    for cd in crates:
        raw = load_samples_for_size(cd, size_label)
        if not raw:
            continue
        xs = trim_and_clip(raw, trim_frac, q_low, q_high)
        if xs:
            series[human_crate_label(cd.name)] = xs

    if not series:
        print(f"[warn] No samples to plot for {size_label}")
        return

    # Determine common grid
    all_vals = [v for xs in series.values() for v in xs]
    xmin, xmax = min(all_vals), max(all_vals)
    pad = max(1e-3, 0.02 * (xmax - xmin))
    grid = [xmin - pad + i * (xmax - xmin + 2 * pad) / 400 for i in range(401)]

    plt.figure(figsize=(max(8, 1.6 * len(series)), 5))
    for label, xs in series.items():
        dens = gaussian_kde(xs, grid)
        plt.fill_between(grid, dens, alpha=0.25, linewidth=0)
        plt.plot(grid, dens, label=label, linewidth=1.5)

    plt.title(f"{group} — duration distribution at {size_label}")
    plt.xlabel("Average time per sample (ms)")
    plt.ylabel("Density (a.u.)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc="best", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote plot: {out_png}")

def plot_combined_box(target_dir: Path, group: str, size_label: str,
                      trim_frac: float, q_low: float, q_high: float,
                      out_png: Path) -> None:
    root = target_dir / "criterion" / group
    crates = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name != "report"]

    labels: List[str] = []
    data: List[List[float]] = []
    for cd in crates:
        raw = load_samples_for_size(cd, size_label)
        if not raw:
            continue
        xs = trim_and_clip(raw, trim_frac, q_low, q_high)
        if xs:
            labels.append(human_crate_label(cd.name))
            data.append(xs)

    if not data:
        print(f"[warn] No samples to plot for {size_label}")
        return

    # Sort by median for nicer display
    order = sorted(range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2])
    labels = [labels[i] for i in order]
    data = [data[i] for i in order]

    plt.figure(figsize=(max(8, 1.2 * len(labels)), 5))
    bp = plt.boxplot(
        data,
        vert=True,
        showfliers=False,   # hide outliers to avoid artefact spikes
        widths=0.6,
        patch_artist=True
    )
    for box in bp['boxes']:
        box.set_alpha(0.25)

    plt.xticks(range(1, len(labels) + 1), labels, rotation=15, ha='right')
    plt.ylabel("Average time per sample (ms)")
    plt.title(f"{group} — box & whiskers at {size_label}")
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote plot: {out_png}")


# ----------------------------- CLI -----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Criterion results and create combined plots.")
    ap.add_argument("--group", default="yaml_parse", help="Criterion group (default: yaml_parse)")
    ap.add_argument("--root", default="target", help="Path to target directory (default: ./target)")
    ap.add_argument("--out-csv", default="summary.csv", help="CSV output path")
    ap.add_argument("--out-md", default="summary.md", help="Markdown output path")
    ap.add_argument("--size", default="25MiB", help="Size label to plot (default: 25MiB)")
    ap.add_argument("--kde-png", default=None, help="Output PNG for KDE (default: combined_kde_<size>.png)")
    ap.add_argument("--box-png", default=None, help="Output PNG for boxplot (default: combined_box_<size>.png)")
    ap.add_argument("--trim", type=float, default=0.25, help="Trim fraction of earliest samples (default: 0.25)")
    ap.add_argument("--q-low", type=float, default=0.10, help="Lower quantile for clipping (default: 0.10)")
    ap.add_argument("--q-high", type=float, default=0.99, help="Upper quantile for clipping (default: 0.99)")
    args = ap.parse_args()

    target_dir = Path(args.root).resolve()
    rows = collect_records(target_dir, args.group)
    if not rows:
        raise SystemExit("[error] No records found. Check --group and ./target/criterion layout")

    # Tables
    write_csv(rows, Path(args.out_csv))
    write_markdown_table(rows, Path(args.out_md))
    print_rankings(rows)

    # Plots
    kde_png = Path(args.kde_png) if args.kde_png else Path(f"combined_kde_{args.size}.png")
    box_png = Path(args.box_png) if args.box_png else Path(f"combined_box_{args.size}.png")

    plot_combined_kde(target_dir, args.group, args.size, args.trim, args.q_low, args.q_high, kde_png)
    plot_combined_box(target_dir, args.group, args.size, args.trim, args.q_low, args.q_high, box_png)

    print(f"\nWrote: {args.out_csv}, {args.out_md}, {kde_png}, {box_png}")

if __name__ == "__main__":
    main()
