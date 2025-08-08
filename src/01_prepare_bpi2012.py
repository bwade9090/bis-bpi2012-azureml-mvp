#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01 — Prepare BPI 2012 (Clean + Label)
- Load BPI 2012 (CSV or XES)
- Normalize schema: case_id, activity, timestamp[, resource]
- Compute case-level features
- Create label_late using q=0.75 of duration_hours
- Save:
    artifacts/processed_case_features.csv
    artifacts/processed_case_features_labeled.csv
"""

import os
import sys
import argparse
import pandas as pd

# src.utils import (both "src.utils" and "utils" fallback)
try:
    from src.utils import ensure_schema, compute_case_features, make_labels
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils import ensure_schema, compute_case_features, make_labels  # type: ignore


def load_event_log(data_dir: str, xes_name: str) -> pd.DataFrame:
    xes_path = os.path.join(data_dir, xes_name)

    if os.path.exists(xes_path):
        print(f"[INFO] Loading XES: {xes_path}")
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py import convert_to_dataframe

        log = xes_importer.apply(xes_path)
        df = convert_to_dataframe(log)
        df = df.rename(columns={
            "case:concept:name": "case_id",
            "concept:name": "activity",
            "time:timestamp": "timestamp",
            "org:resource": "resource"
        })
        df = ensure_schema(df, case_col="case_id", act_col="activity", ts_col="timestamp", res_col="resource")
        return df

    raise FileNotFoundError(
        f"Could not find either CSV({csv_path}) or XES({xes_path}). "
        "Put your file in the data/ folder. See data/README.md."
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare BPI2012 features and labels (Day 1).")
    parser.add_argument("--data-dir", default="data", help="Directory containing CSV/XES")
    parser.add_argument("--xes-name", default="BPI_Challenge_2012.xes", help="XES filename")
    parser.add_argument("--processed-path", default="artifacts/processed_case_features.csv", help="Output features CSV")
    parser.add_argument("--labeled-path", default="artifacts/processed_case_features_labeled.csv", help="Output labeled CSV")
    parser.add_argument("--quantile", type=float, default=0.75, help="Quantile for 'late' threshold (default 0.75)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.processed_path), exist_ok=True)

    df = load_event_log(args.data_dir, args.xes_name)
    print(f"[INFO] Loaded events: {len(df):,}")

    features = compute_case_features(df)
    labeled, thr = make_labels(features, quantile=args.quantile)

    features.to_csv(args.processed_path, index=False)
    labeled.to_csv(args.labeled_path, index=False)

    print(f"[OK] Saved features -> {args.processed_path}")
    print(f"[OK] Saved labeled -> {args.labeled_path}")
    print(f"[METRIC] Late threshold (duration_hours q{args.quantile:.2f}): {thr:.4f}")


if __name__ == "__main__":
    main()
