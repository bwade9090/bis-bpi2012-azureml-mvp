#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02 — Process Mining EDA (Variants / Bottlenecks)
- Load BPI2012 (CSV/XES)
- Basic case stats (counts, median case length)
- Build and save DFG (frequency)
- Save EDA outputs to artifacts/eda/
"""

import os
import sys
import argparse
import pandas as pd

def load_event_log(data_dir: str, xes_name: str) -> pd.DataFrame:
    xes_path = os.path.join(data_dir, xes_name)

    if os.path.exists(xes_path):
        print(f"[INFO] Loading XES: {xes_path}")
        from pm4py import read_xes
        log = read_xes(xes_path)
        log = log.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])
        return log

    raise FileNotFoundError(
        f"Could not find XES({xes_path}). "
        "Put your file in the data/ folder. See data/README.md."
    )

def print_stat(log: pd.DataFrame):
    case_lengths = log.groupby("case:concept:name").size()
    print(f"[STAT] Cases: {case_lengths.shape[0]:,}")
    print(f"[STAT] Events: {log.shape[0]:,}")
    print(f"[STAT] Median case length: {int(case_lengths.median())}")

def save_dfg_image(log: pd.DataFrame, out_path: str):
    # PM4Py DFG (frequency) visualization — requires graphviz. If unavailable, save edges as CSV fallback.
    print("[INFO] Building DFG (frequency)...")
    
    from pm4py import discover_dfg, save_vis_dfg
    dfg, start_activities, end_activities = discover_dfg(log)
    
    d = os.path.dirname(out_path)
    os.makedirs(d, exist_ok=True)
    save_vis_dfg(dfg, start_activities, end_activities, file_path=out_path)
    print(f"[OK] Saved DFG image -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process Mining EDA for BPI2012 (Day 1).")
    parser.add_argument("--data-dir", default="data", help="Directory containing XES")
    parser.add_argument("--xes-name", default="BPI_Challenge_2012.xes")
    parser.add_argument("--out-image", default="artifacts/eda/dfg_frequency.png")
    args = parser.parse_args()

    log = load_event_log(args.data_dir, args.xes_name)

    # Basic stats
    print_stat(log)

    # DFG visualization
    save_dfg_image(log, args.out_image)


if __name__ == "__main__":
    main()