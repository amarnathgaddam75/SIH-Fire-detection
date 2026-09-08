"""
export_unknown_rows.py
======================
Filters data/feature_vector.csv to rows where class_label == "Unknown",
exports a clean CSV to data/unknown_rows_review.csv containing only:
  latitude, longitude, acq_date, frp, confidence, type, distinct_days,
  distance_to_industrial_m, distance_to_farmland_m, frp_deviation.

Prints these rows directly to the console in a readable format, sorted by
distinct_days descending, with a one-line failure condition explanation for each.
"""

import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "feature_vector.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "unknown_rows_review.csv")

TARGET_COLS = [
    "latitude",
    "longitude",
    "acq_date",
    "frp",
    "confidence",
    "type",
    "distinct_days",
    "distance_to_industrial_m",
    "distance_to_farmland_m",
    "frp_deviation",
]


def explain_failure(row):
    """Generate a one-line note explaining which specific classification condition failed."""
    dist_ind = row["distance_to_industrial_m"]
    dist_farm = row["distance_to_farmland_m"]
    h_type = row["type"]
    days = row["distinct_days"]

    # Check proximity to industrial
    if dist_ind <= 1000:
        if pd.isna(h_type):
            type_str = "type=NaN"
        else:
            type_str = f"type={int(h_type)}"

        if days >= 8:
            return (
                f"{type_str}, inside industrial buffer ({dist_ind:.1f}m <= 1000m), "
                f"only {int(days)} distinct days -- just under the 10-day persistence threshold (borderline Industrial)"
            )
        else:
            return (
                f"{type_str}, inside industrial buffer ({dist_ind:.1f}m <= 1000m), "
                f"only {int(days)} distinct days (< 10 threshold) -- excluded from Natural Fire/Ag due to industrial proximity"
            )
    else:
        # Outside industrial buffer (> 1000m)
        if pd.isna(h_type):
            return (
                f"type=NaN (missing type), outside industrial buffer ({dist_ind:.1f}m > 1000m) -- "
                f"failed Natural Fire (requires type=0) and failed Ag Burning ({dist_farm:.1f}m > 2000m)"
            )
        elif h_type != 0:
            return (
                f"type={int(h_type)} (non-vegetation), outside industrial buffer ({dist_ind:.1f}m > 1000m) -- "
                f"failed Natural Fire (requires type=0) and failed Ag Burning ({dist_farm:.1f}m > 2000m)"
            )
        else:
            # type == 0 and dist_ind > 1000
            return (
                f"type=0, outside industrial buffer ({dist_ind:.1f}m > 1000m) -- "
                f"unresolved condition (dist_farm={dist_farm:.1f}m, distinct_days={int(days)})"
            )


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print("=" * 80)
    print("GeoFlare -- Unknown Rows Review & Export")
    print("=" * 80)
    print(f"Total rows in {os.path.basename(INPUT_CSV)}: {len(df):,}")

    # Filter to Unknown rows
    unknown_df = df[df["class_label"] == "Unknown"].copy()
    num_unknown = len(unknown_df)
    print(f"Total 'Unknown' rows found: {num_unknown:,}")

    if num_unknown == 0:
        print("No Unknown rows found. Exiting.")
        return

    # Export clean CSV with target columns only
    export_df = unknown_df[TARGET_COLS].copy()
    export_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved clean export ({num_unknown} rows, {len(TARGET_COLS)} columns) to:")
    print(f"  --> {OUTPUT_CSV}\n")

    # Sort descending by distinct_days so borderline cases appear first
    export_df["sort_days"] = export_df["distinct_days"].fillna(0)
    export_df["sort_frp"] = export_df["frp"].fillna(0)
    sorted_df = export_df.sort_values(
        by=["sort_days", "sort_frp"], ascending=[False, False]
    ).drop(columns=["sort_days", "sort_frp"])

    print("=" * 80)
    print(f"BORDERLINE & UNRESOLVED CASES (Sorted by distinct_days DESCENDING)")
    print("=" * 80)

    for rank, (idx, row) in enumerate(sorted_df.iterrows(), start=1):
        note = explain_failure(row)
        lat = f"{row['latitude']:.5f}"
        lon = f"{row['longitude']:.5f}"
        date = str(row['acq_date'])
        frp = f"{row['frp']:.1f}" if pd.notna(row['frp']) else "N/A"
        conf = f"{row['confidence']}" if pd.notna(row['confidence']) else "N/A"
        type_val = f"{int(row['type'])}" if pd.notna(row['type']) else "NaN"
        days = f"{int(row['distinct_days'])}" if pd.notna(row['distinct_days']) else "0"
        dist_ind = f"{row['distance_to_industrial_m']:.1f}m" if pd.notna(row['distance_to_industrial_m']) else "N/A"
        dist_farm = f"{row['distance_to_farmland_m']:.1f}m" if pd.notna(row['distance_to_farmland_m']) else "N/A"
        frp_dev = f"{row['frp_deviation']:.2f}" if pd.notna(row['frp_deviation']) else "N/A"

        print(f"[{rank:2d}/{num_unknown}] Date: {date} | Coords: ({lat}, {lon}) | Days: {days:>2} | FRP: {frp:>5} MW | Conf: {conf:>3}")
        print(f"     Type: {type_val:<3} | Dist_Ind: {dist_ind:<9} | Dist_Farm: {dist_farm:<10} | FRP_Dev: {frp_dev}")
        print(f"     NOTE: {note}")
        print("-" * 80)

    print("\nSummary Statistics for Unknown Rows:")
    print(f"  - Total Unknown rows: {num_unknown}")
    print(f"  - Max distinct_days: {sorted_df['distinct_days'].max():.0f}")
    print(f"  - Rows near industrial (<= 1000m): {(sorted_df['distance_to_industrial_m'] <= 1000).sum()}")
    print(f"  - Rows with missing type (NaN): {sorted_df['type'].isna().sum()}")
    print(f"  - Output file written: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
