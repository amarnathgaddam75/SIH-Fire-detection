"""
compute_escalation.py
=====================
Early-Warning Thermal Escalation Analysis for GeoFlare.

Problem:
FIRMS fires and industrial flare sources can escalate rapidly before crossing
the traditional static anomaly threshold (e.g., FRP deviation > 2.5x cluster baseline).
Waiting for a 2.5x spike means an alert is only triggered *after* an uncontrolled
flare or blowout has already reached critical scale.

This script implements an explainable early-warning trend detector that analyzes
the velocity and momentum of thermal intensity across recent satellite passes.

Pipeline:
1. For each cluster_id (excluding noise points, cluster_id != -1):
   - Sort detections chronologically by acquisition date and time.
   - Look at the most recent 3-5 detections (or whatever is available if < 5).
2. Trend Detection Logic:
   - Fit a 1st-degree linear regression polynomial y = m*x + c across the window.
   - Count consecutive increases: Delta_i = FRP_i - FRP_{i-1} > 0.
   - Check Net Direction: FRP_last > FRP_first.
   - A cluster is classified as "Escalating" if:
       a) Strictly increasing: Every step in the window has Delta_i > 0.
       b) Consecutive increases: At least 3 consecutive upward steps.
       c) Positive regression slope: Linear regression slope m > 0.10 MW/pass,
          with net positive gain and at least half the steps positive.
       d) 2-pass surge: For small clusters with only 2 detections, a positive surge
          where FRP increases with slope > 0.50 MW.
3. Categorization into `escalation_status`:
   - "Abnormal": Row has already crossed the 2.5x baseline threshold (alert_flag == 'Abnormal spike').
                 Keeps existing alert_flag logic completely untouched.
   - "Escalating": Row belongs to the active recent window of an escalating cluster,
                   trending upward but not yet past the 2.5x abnormal threshold.
   - "Stable": Neither escalating nor abnormal.
4. Reports summary counts and prints a detailed sanity-check table of all "Escalating"
   detections along with their 5-pass FRP reading history.
5. Saves the updated dataset back to data/feature_vector.csv and exports `escalation_status`
   to data/firms_final_ml.geojson and data/firms_final.geojson.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
GEOJSON_ML_PATH = os.path.join(DATA_DIR, "firms_final_ml.geojson")
GEOJSON_FINAL_PATH = os.path.join(DATA_DIR, "firms_final.geojson")


def evaluate_cluster_trend(frps):
    """
    Evaluates thermal momentum and trend across recent FRP readings.

    Parameters:
        frps (list of float): Sequence of 2 to 5 consecutive FRP readings.

    Returns:
        is_escalating (bool): True if thermal trend is actively rising.
        slope (float): Linear regression slope in MW per pass.
        rule_description (str): Human-readable explanation of the trend trigger.
    """
    k = len(frps)
    if k < 2:
        return False, 0.0, "Insufficient observations (<2 detections)"

    x = np.arange(k)
    slope, _ = np.polyfit(x, frps, 1)
    diffs = [frps[i] - frps[i - 1] for i in range(1, k)]
    pos_diffs = sum(1 for d in diffs if d > 0)
    net_inc = frps[-1] > frps[0]

    # Calculate max consecutive increases
    cons = 0
    max_cons = 0
    for d in diffs:
        if d > 0:
            cons += 1
            max_cons = max(max_cons, cons)
        else:
            cons = 0

    # Rule 1: Strictly increasing across all observations in window
    if all(d > 0 for d in diffs) and net_inc:
        return True, slope, f"Strictly increasing across all {k} passes (all deltas > 0)"

    # Rule 2: At least 3 consecutive increases
    if max_cons >= 3 and net_inc:
        return True, slope, f"Sustained momentum ({max_cons} consecutive increases)"

    # Rule 3: Mostly increasing with strong positive slope (k >= 3)
    if k >= 3 and slope > 0.10 and net_inc and (pos_diffs >= len(diffs) - 1):
        return True, slope, f"Mostly increasing (slope = +{slope:.2f} MW/pass, {pos_diffs}/{len(diffs)} increases)"

    # Rule 4: Positive linear regression slope where final reading is at or above mean (k >= 3)
    if k >= 3 and slope > 0.20 and net_inc and frps[-1] >= np.mean(frps):
        return True, slope, f"Positive linear regression slope (+{slope:.2f} MW/pass, net +{frps[-1]-frps[0]:.1f} MW)"

    # Rule 5: 2-point surge for small clusters (k == 2)
    if k == 2 and slope > 0.50 and net_inc:
        return True, slope, f"Surging 2-pass sequence (+{slope:.2f} MW surge)"

    return False, slope, "Stable thermal pattern"


def main():
    print("=" * 80)
    print("GeoFlare -- Thermal Escalation Analysis & Early Warning Detection")
    print("=" * 80)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Feature vector CSV not found: {CSV_PATH}")

    # 1. Load dataset
    print(f"\n1. Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Loaded {len(df):,} rows x {len(df.columns)} columns.")

    # Sort chronologically by acquisition date and time to establish true time-series
    time_col = "acq_time" if "acq_time" in df.columns else None
    if time_col:
        df["_sort_time"] = pd.to_numeric(df[time_col], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(by=["acq_date", "_sort_time"]).reset_index(drop=True)
        df.drop(columns=["_sort_time"], inplace=True)
    else:
        df = df.sort_values(by=["acq_date"]).reset_index(drop=True)

    # 2. Group by cluster and analyze recent 3-5 detections
    print("\n2. Analyzing recent 3-5 detections per spatial cluster...")
    clusters = df[df["cluster_id"] != -1]["cluster_id"].unique()
    print(f"   Evaluating {len(clusters)} distinct persistent clusters (excluding noise)...")

    escalating_cluster_map = {}  # cid -> dict of trend info
    recent_window_indices = set()
    cluster_recent_frps = {}     # cid -> list of last 5 frps

    for cid in sorted(clusters):
        c_sub = df[df["cluster_id"] == cid]
        n_obs = len(c_sub)

        # Most recent 3-5 detections (or whatever is available if < 5)
        window_size = min(5, n_obs)
        window = c_sub.tail(window_size)
        frps = [float(v) for v in window["frp"].tolist()]
        cluster_recent_frps[cid] = frps

        is_esc, slope, rule = evaluate_cluster_trend(frps)

        if is_esc:
            escalating_cluster_map[cid] = {
                "cluster_id": cid,
                "total_obs": n_obs,
                "window_size": window_size,
                "window_indices": window.index.tolist(),
                "recent_frps": frps,
                "slope": slope,
                "rule": rule,
                "dates": window["acq_date"].tolist(),
                "latest_date": window.iloc[-1]["acq_date"],
                "latest_frp": frps[-1]
            }
            # Track the recent window rows
            recent_window_indices.update(window.index.tolist())

    print(f"   Identified {len(escalating_cluster_map)} escalating clusters exhibiting upward thermal momentum.")

    # 3. Assign escalation_status column
    print("\n3. Assigning escalation_status categories...")
    # Default to "Stable"
    df["escalation_status"] = "Stable"

    # Rule A: Keep existing alert_flag logic untouched: any Abnormal spike -> "Abnormal"
    is_abnormal = df["alert_flag"] == "Abnormal spike"
    df.loc[is_abnormal, "escalation_status"] = "Abnormal"

    # Rule B: Rows in the active escalating window of escalating clusters (not already Abnormal) -> "Escalating"
    for cid, info in escalating_cluster_map.items():
        for idx in info["window_indices"]:
            if df.loc[idx, "escalation_status"] != "Abnormal":
                df.loc[idx, "escalation_status"] = "Escalating"

    # 4. Print Distribution & Sanity Check Table
    print("\n" + "-" * 45)
    print("  ESCALATION STATUS DISTRIBUTION (Row Counts)")
    print("-" * 45)
    counts = df["escalation_status"].value_counts()
    for cat in ["Abnormal", "Escalating", "Stable"]:
        cnt = counts.get(cat, 0)
        pct = (cnt / len(df)) * 100
        print(f"    {cat:15s} : {cnt:5,d}  ({pct:5.1f}%)")
    print("-" * 45)
    print(f"    {'TOTAL':15s} : {len(df):5,d}  (100.0%)\n")

    # Sanity Check Table: List all "Escalating" rows with their last 5 FRP readings
    print("=" * 115)
    print("SANITY CHECK: ESCALATING DETECTIONS & LAST 3-5 FRP READINGS")
    print("=" * 115)
    print(f"{'Row ID':<7} {'Cluster':<8} {'Date':<11} {'FRP (MW)':<10} {'FRP Dev':<9} {'Slope':<8} {'Last 3-5 FRP Readings':<30} {'Trend Trigger':<25}")
    print("-" * 115)

    escalating_rows = df[df["escalation_status"] == "Escalating"]

    for idx, row in escalating_rows.iterrows():
        cid = int(row["cluster_id"])
        c_info = escalating_cluster_map.get(cid, {})
        frp_history = c_info.get("recent_frps", [row["frp"]])
        frp_hist_str = "[" + ", ".join(f"{v:.1f}" for v in frp_history) + "]"
        slope_val = c_info.get("slope", 0.0)
        rule_val = c_info.get("rule", "Trending up")
        frp_dev_val = f"{row['frp_deviation']:.2f}x" if pd.notnull(row["frp_deviation"]) else "N/A"

        print(f"{idx:<7d} {cid:<8d} {str(row['acq_date']):<11} {row['frp']:<10.2f} {frp_dev_val:<9} {slope_val:<+8.2f} {frp_hist_str:<30} {rule_val[:24]:<25}")

    print("=" * 115)
    print(f"Total Escalating detections displayed: {len(escalating_rows)}")
    print(f"Across {len(escalating_cluster_map)} distinct thermal clusters.")

    # 5. Save back to data/feature_vector.csv
    print(f"\n5. Saving updated data/feature_vector.csv...")
    df.to_csv(CSV_PATH, index=False)
    print(f"   Saved {len(df):,} rows x {len(df.columns)} columns to {CSV_PATH}.")

    # 6. Re-export escalation_status to GeoJSONs
    # Update data/firms_final_ml.geojson
    if os.path.exists(GEOJSON_ML_PATH):
        print(f"\n6a. Syncing escalation_status to {GEOJSON_ML_PATH}...")
        with open(GEOJSON_ML_PATH, "r") as f:
            geojson_data = json.load(f)

        features = geojson_data.get("features", [])
        matched = 0
        for i, feat in enumerate(features):
            if i < len(df):
                cid = int(df.iloc[i]["cluster_id"]) if pd.notnull(df.iloc[i]["cluster_id"]) else -1
                feat["properties"]["cluster_id"] = cid
                feat["properties"]["escalation_status"] = str(df.iloc[i]["escalation_status"])
                if cid in cluster_recent_frps:
                    feat["properties"]["recent_frps"] = cluster_recent_frps[cid]
                matched += 1

        with open(GEOJSON_ML_PATH, "w") as f:
            json.dump(geojson_data, f, indent=2)
        print(f"    Synced {matched:,} features in firms_final_ml.geojson.")

    # Update data/firms_final.geojson
    if os.path.exists(GEOJSON_FINAL_PATH):
        print(f"\n6b. Syncing escalation_status to {GEOJSON_FINAL_PATH}...")
        with open(GEOJSON_FINAL_PATH, "r") as f:
            base_geojson = json.load(f)

        features = base_geojson.get("features", [])
        matched = 0
        for i, feat in enumerate(features):
            if i < len(df):
                cid = int(df.iloc[i]["cluster_id"]) if pd.notnull(df.iloc[i]["cluster_id"]) else -1
                feat["properties"]["cluster_id"] = cid
                feat["properties"]["escalation_status"] = str(df.iloc[i]["escalation_status"])
                if cid in cluster_recent_frps:
                    feat["properties"]["recent_frps"] = cluster_recent_frps[cid]
                matched += 1

        with open(GEOJSON_FINAL_PATH, "w") as f:
            json.dump(base_geojson, f, indent=2)
        print(f"    Synced {matched:,} features in firms_final.geojson.")

    print("\n" + "=" * 80)
    print("Escalation analysis complete! All artifacts synchronized.")
    print("=" * 80)


if __name__ == "__main__":
    main()
