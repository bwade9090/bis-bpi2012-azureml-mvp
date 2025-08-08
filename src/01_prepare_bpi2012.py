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


def load_event_log(data_dir: str, xes_name: str) -> pd.DataFrame:
    xes_path = os.path.join(data_dir, xes_name)

    if os.path.exists(xes_path):
        print(f"[INFO] Loading XES: {xes_path}")
        from pm4py import read_xes, convert_to_dataframe

        log = read_xes(xes_path)
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
        f"Could not find XES({xes_path}). "
        "Put your file in the data/ folder. See data/README.md."
    )

def ensure_schema(df, case_col, act_col, ts_col, res_col=None):
    # normalize column names
    df = df.rename(columns={
        case_col: "case_id",
        act_col: "activity",
        ts_col: "timestamp",
        **({res_col: "resource"} if res_col and res_col in df.columns else {})
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["case_id", "activity", "timestamp"])
    return df

def compute_case_features(df: pd.DataFrame) -> pd.DataFrame:
    # assumes columns: case_id, activity, timestamp, [resource]
    df = df.sort_values(["case_id", "timestamp"])
    grp = df.groupby("case_id")
    first_ts = grp["timestamp"].min()
    last_ts = grp["timestamp"].max()
    case_length = grp.size()
    n_resources = grp["resource"].nunique() if "resource" in df.columns else pd.Series(0, index=case_length.index)

    # inter-event times per case (in minutes)
    def inter_event_minutes(g):
        g = g.sort_values("timestamp")
        delta = g["timestamp"].diff().dt.total_seconds().div(60).dropna()
        if len(delta) == 0:
            return pd.Series({"mean_inter_event_minutes": 0.0, "max_inter_event_minutes": 0.0})
        return pd.Series({"mean_inter_event_minutes": float(delta.mean()), "max_inter_event_minutes": float(delta.max())})

    inter = grp.apply(inter_event_minutes)

    # weekend events count
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    n_weekend = grp["is_weekend"].sum()

    # first/last event hour
    first_hour = grp["timestamp"].min().dt.hour
    last_hour = grp["timestamp"].max().dt.hour

    # duration (hours)
    duration_hours = (last_ts - first_ts).dt.total_seconds().div(3600)

    features = pd.DataFrame({
        "case_id": case_length.index,
        "case_length": case_length.values,
        "n_resources": n_resources.values,
        "mean_inter_event_minutes": inter["mean_inter_event_minutes"].values,
        "max_inter_event_minutes": inter["max_inter_event_minutes"].values,
        "n_weekend_events": n_weekend.values,
        "first_event_hour": first_hour.values,
        "last_event_hour": last_hour.values,
        "duration_hours": duration_hours.values
    })
    return features

def make_labels(features: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
    thr = features["duration_hours"].quantile(quantile)
    labels = (features["duration_hours"] > thr).astype(int).rename("label_late")
    return pd.concat([features.drop(columns=["duration_hours"]), labels], axis=1), float(thr)


def main():
    parser = argparse.ArgumentParser(description="Prepare BPI2012 features and labels (Day 1).")
    parser.add_argument("--data-dir", default="data", help="Directory containing XES")
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
