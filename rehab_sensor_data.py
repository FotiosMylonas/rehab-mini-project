# rehab_sensor_dat.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.errors import ParserError
from io_utils import unzip_all, pick_csv
from signal_processing import (
    infer_time_and_fs,
    choose_signal,
    butter_lowpass_filter,
    detect_steps,
)


def read_sensor_csv(path):
    """
    this function reads IMU CSV files that may contain metadata lines before the real table.
    Tries to auto-detect delimiter and skip the metadata.
    """
    # First attempt: normal read
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except ParserError:
        pass

    # Read a chunk of the top of the file
    seps = [",", ";", "\t"]
    lines = []
    with open(path, "r", errors="ignore") as f:
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            if line.strip():
                lines.append(line.rstrip("\n"))

    # Choose delimiter by total counts
    sep_score = {s: 0 for s in seps}
    for ln in lines:
        if ln.lstrip().startswith("#"):
            continue
        for s in seps:
            sep_score[s] += ln.count(s)
    sep = max(sep_score, key=sep_score.get)

    # Determine number of columns 
    field_counts = []
    for ln in lines:
        if ln.lstrip().startswith("#"):
            continue
        c = ln.count(sep) + 1
        if c >= 5:  # ignore tiny metadata lines
            field_counts.append(c)

    if not field_counts:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

    mode_cols = max(set(field_counts), key=field_counts.count)

    # i find where the table starts
    start_row = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#"):
            continue
        if (ln.count(sep) + 1) == mode_cols:
            start_row = i
            break

    # Read from start_row onward
    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        skiprows=start_row,
        on_bad_lines="skip",
    )

    
    def looks_numeric(s):
        try:
            float(str(s))
            return True
        except:
            return False

    if len(df.columns) > 0:
        numeric_header_ratio = sum(looks_numeric(c) for c in df.columns) / len(df.columns)
        if numeric_header_ratio > 0.6:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                skiprows=start_row,
                header=None,
                on_bad_lines="skip",
            )

    return df   


def main():
    parser = argparse.ArgumentParser(description="Rehab mini project: filter + peak-based step counting.")
    parser.add_argument("--zips", type=str, default="data_zips")
    parser.add_argument("--extract", type=str, default="data_extracted")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--fallback_fs", type=float, default=100.0)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    zips_dir = Path(args.zips)
    extract_dir = Path(args.extract)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    unzip_all(zips_dir, extract_dir)
    csv_path = pick_csv(extract_dir)
    print(f"\nUsing CSV file:\n  {csv_path}\n")

    df = read_sensor_csv(csv_path)

    print("Loaded shape:", df.shape)
    print("First columns:", list(df.columns)[:10])
    print(df.head(3))

    time_s, fs = infer_time_and_fs(df, fallback_fs=args.fallback_fs)
    signal, signal_label = choose_signal(df)
    signal = pd.Series(signal).interpolate(limit_direction="both").to_numpy()

    filtered = butter_lowpass_filter(signal, cutoff=args.cutoff, fs=fs)
    peaks, metrics = detect_steps(filtered, fs=fs, time_s=time_s)

    payload = {
        "csv_file": str(csv_path),
        "signal_used": signal_label,
        "fs_hz": fs,
        "cutoff_hz": args.cutoff,
        **metrics,
    }

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Summary ===")
    for k, v in payload.items():
        print(f"{k}: {v}")

    # Plot first 20s (or fallback)
    if len(time_s) > 1:
        max_t = min(20.0, float(time_s[-1]))
        N = int(np.searchsorted(time_s, max_t, side="right"))
        N = max(N, 1)
    else:
        N = min(2000, len(signal))

    peaks_win = peaks[peaks < N]

    plt.figure(figsize=(12, 6))
    plt.plot(time_s[:N], signal[:N], alpha=0.5, label=f"Raw ({signal_label})")
    plt.plot(time_s[:N], filtered[:N], linewidth=2, label=f"Filtered (lowpass {args.cutoff} Hz)")
    plt.scatter(time_s[peaks_win], filtered[peaks_win], s=35, label="Detected steps")
    plt.title("Rehab Mini Project — Step Counting & Movement Frequency")
    plt.xlabel("Time (s)")
    plt.ylabel("Sensor signal (a.u.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "steps_plot.png", dpi=200)
    plt.show()

    print(f"\nSaved:\n  {out_dir / 'summary_metrics.json'}\n  {out_dir / 'steps_plot.png'}")


if __name__ == "__main__":
    main()