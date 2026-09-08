"""
reclassify_ag_burn.py — Reclassify with relaxed Agricultural Burning threshold
===============================================================================
Loads existing feature_vector.csv (which already has distance columns),
re-runs ONLY the 5-class classification logic with distance_to_farmland_m
<= 2000m (instead of 1000m) for Agricultural Burning.

If count is still zero at 2000m, tries 3000m.
Prints the min distance_to_farmland_m for diagnostics.
Overwrites ONLY the class_label column in feature_vector.csv.
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} rows from feature_vector.csv")
print(f"\nMin distance_to_farmland_m across entire dataset: "
      f"{df['distance_to_farmland_m'].min():.1f} m")
print(f"Max distance_to_farmland_m: {df['distance_to_farmland_m'].max():.1f} m")
print(f"Median distance_to_farmland_m: {df['distance_to_farmland_m'].median():.1f} m")

# Show distribution of farmland distances
print(f"\nRows within various farmland distance thresholds:")
for thresh in [500, 1000, 1500, 2000, 2500, 3000, 5000]:
    n = (df["distance_to_farmland_m"] <= thresh).sum()
    print(f"  <= {thresh}m: {n} rows")


def run_classification(df, ag_threshold):
    """Re-run the 5-class classification with a given ag burn threshold."""
    df["class_label"] = "Unknown"

    # Rule 1: Industrial (unchanged)
    near_industrial = df["distance_to_industrial_m"] <= 1000
    type_is_2 = df["type"] == 2
    high_persistence = df["distinct_days"] >= 10
    industrial_rule = near_industrial & (type_is_2 | high_persistence)
    df.loc[industrial_rule, "class_label"] = "Industrial"

    # Rule 2: Agricultural Burning (RELAXED threshold)
    near_farmland = df["distance_to_farmland_m"] <= ag_threshold
    not_near_industrial = df["distance_to_industrial_m"] > 1000
    low_persistence = df["distinct_days"] <= 3
    ag_burn_rule = near_farmland & not_near_industrial & low_persistence
    still_unknown = df["class_label"] == "Unknown"
    df.loc[still_unknown & ag_burn_rule, "class_label"] = "Agricultural Burning"

    # Rule 3: Natural Fire (type=0 AND not near industrial)
    type_is_0 = df["type"] == 0
    still_unknown = df["class_label"] == "Unknown"
    df.loc[still_unknown & type_is_0 & not_near_industrial, "class_label"] = "Natural Fire"

    # Rule 4: Unknown — everything else stays
    return df


# --- Try 2000m threshold ---
print("\n" + "=" * 72)
print(f"CLASSIFICATION WITH distance_to_farmland_m <= 2000m")
print("=" * 72)

df = run_classification(df, ag_threshold=2000)

print("\n  class_label breakdown:")
print("  " + "-" * 45)
for label, count in df["class_label"].value_counts().sort_index().items():
    print(f"    {label:25s} : {count:,}")
print("  " + "-" * 45)
print(f"    {'TOTAL':25s} : {len(df):,}")

ag_rows = df[df["class_label"] == "Agricultural Burning"]
print(f"\n  Agricultural Burning rows: {len(ag_rows)}")

if len(ag_rows) > 0:
    print("\n  Details of Agricultural Burning rows:")
    print(ag_rows[["latitude", "longitude", "acq_date", "frp",
                    "distinct_days", "distance_to_farmland_m"]].to_string(index=False))
    CHOSEN_THRESHOLD = 2000
else:
    # --- Try 3000m threshold ---
    print("\n  Still zero at 2000m — trying 3000m...")
    print("\n" + "=" * 72)
    print(f"CLASSIFICATION WITH distance_to_farmland_m <= 3000m")
    print("=" * 72)

    df = run_classification(df, ag_threshold=3000)

    print("\n  class_label breakdown:")
    print("  " + "-" * 45)
    for label, count in df["class_label"].value_counts().sort_index().items():
        print(f"    {label:25s} : {count:,}")
    print("  " + "-" * 45)
    print(f"    {'TOTAL':25s} : {len(df):,}")

    ag_rows = df[df["class_label"] == "Agricultural Burning"]
    print(f"\n  Agricultural Burning rows: {len(ag_rows)}")

    if len(ag_rows) > 0:
        print("\n  Details of Agricultural Burning rows:")
        print(ag_rows[["latitude", "longitude", "acq_date", "frp",
                        "distinct_days", "distance_to_farmland_m"]].to_string(index=False))
    CHOSEN_THRESHOLD = 3000

# --- Save updated CSV (overwrite class_label only) ---
print("\n" + "=" * 72)
print("Saving updated class_label to feature_vector.csv")
print("=" * 72)
df.to_csv(CSV_PATH, index=False)
print(f"  Saved {len(df):,} rows to {CSV_PATH}")
print(f"  (Only class_label column was recomputed; all other columns unchanged)")
