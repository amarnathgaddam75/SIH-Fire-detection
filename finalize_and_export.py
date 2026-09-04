"""
finalize_and_export.py  —  Post-Classification Finalization & GeoJSON Export
=============================================================================
This script picks up where classify_hotspots.py left off.  It reads the
labeled CSV and produces a clean GeoJSON file ready for the interactive
Leaflet dashboard.

What it does, step by step:
  1. Loads data/firms_labeled.csv (the output of classify_hotspots.py)
  2. Creates a human-readable "final_label" column:
       - "Industrial"   if the row was tagged industrial_direct or industrial_nearby
       - "Natural Fire"  if the row is unclassified AND NASA type == 0 (vegetation)
       - For remaining unclassified rows (NRT data, type unknown):
           "Industrial"   if its cluster was active >= 10 distinct days
           "Natural Fire"  otherwise
  3. Computes "frp_deviation" — how far each point's Fire Radiative Power
     is from its cluster average (NaN for noise points not in any cluster)
  4. Sets "alert_flag" to "Abnormal spike" if frp_deviation > 2.5x
  5. Builds a one-sentence "evidence" summary for each row
  6. Exports as GeoJSON (data/firms_final.geojson) with Point features

Dependencies: pandas, geopandas, shapely
Install:      pip install pandas geopandas shapely
"""

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD THE LABELED CSV
# ──────────────────────────────────────────────────────────────────────────────
# classify_hotspots.py saved this file with columns like:
#   latitude, longitude, frp, type, label, cluster_id, distinct_days, ...
# We load it back into a DataFrame and then convert to a GeoDataFrame
# so we can export as GeoJSON later.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "data", "firms_labeled.csv")

print("=" * 72)
print("STEP 1: Loading data/firms_labeled.csv")
print("=" * 72)

df = pd.read_csv(INPUT_PATH)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 2. CREATE "final_label"
# ──────────────────────────────────────────────────────────────────────────────
# The "label" column from classify_hotspots.py has three values:
#   - "industrial_direct"  -> the hotspot fell directly inside an OSM industrial zone
#   - "industrial_nearby"  -> the hotspot was within 1 km of an OSM industrial zone
#   - "unclassified"       -> no OSM match was found
#
# We now collapse those into a simple binary classification:
#   "Industrial"  vs  "Natural Fire"
#
# For unclassified rows, we use two tiebreakers:
#   a) NASA's "type" column: type=0 means "presumed vegetation fire" -> Natural Fire
#   b) Persistence: if the cluster was seen on >=10 different days, it's almost
#      certainly a permanent heat source (factory, power plant) -> Industrial
#
# Why >=10 days?  A forest fire rarely stays in the exact same spot for more
# than a few days.  A coal plant runs 365 days/year.

print("STEP 2: Assigning final_label")

# Start with "Unknown" so we can verify every row gets classified.
df["final_label"] = "Unknown"

# Rule 1: Any row already labeled as industrial -> "Industrial"
industrial_mask = df["label"].isin(["industrial_direct", "industrial_nearby"])
df.loc[industrial_mask, "final_label"] = "Industrial"

# Rule 2: Unclassified rows where NASA says type=0 -> "Natural Fire"
# type=0 means NASA's own algorithm believes this is a vegetation fire.
unclassified_mask = df["label"] == "unclassified"
type_is_zero = df["type"] == 0
df.loc[unclassified_mask & type_is_zero, "final_label"] = "Natural Fire"

# Rule 3: Remaining unclassified rows (type is NaN, i.e. NRT data
# where NASA hasn't assigned a type yet).  We fall back to our own
# persistence heuristic: >=10 distinct days -> Industrial.
#
# Why does this work?  Because NRT data is recent satellite data that
# NASA hasn't fully processed yet, so the "type" column is missing.
# But our DBSCAN clustering + day-counting already ran on these points
# in classify_hotspots.py.  If a point belongs to a cluster that was
# detected on 10+ different days, the simplest explanation is that
# it's a permanent industrial heat source.
still_unknown = df["final_label"] == "Unknown"
persistent_mask = df["distinct_days"] >= 10
df.loc[still_unknown & persistent_mask, "final_label"] = "Industrial"
df.loc[still_unknown & ~persistent_mask, "final_label"] = "Natural Fire"

# Sanity check: no row should be "Unknown" anymore.
assert (df["final_label"] == "Unknown").sum() == 0, "Some rows were not classified!"

print("  final_label counts:")
for label, count in df["final_label"].value_counts().items():
    print(f"    {label:20s} : {count:,}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. CREATE "frp_deviation"
# ──────────────────────────────────────────────────────────────────────────────
# FRP = Fire Radiative Power, measured in megawatts.  It tells you how
# intensely a hotspot is burning.  A forest fire might be 5-50 MW; a
# steel smelter might be 200 MW.
#
# "frp_deviation" answers: "How many times the cluster average is this
# particular reading?"  If a cluster normally reads 10 MW and one
# reading is 30 MW, its frp_deviation = 30/10 = 3.0x -- an anomaly.
#
# Points not in any cluster (cluster_id == -1, "noise") get NaN because
# there's no cluster average to compare against.

print("STEP 3: Computing frp_deviation")

# Calculate the mean FRP for each cluster.
# .transform("mean") broadcasts the group mean back to every row in that group,
# so every point in cluster 7 gets cluster 7's mean FRP.
cluster_mean_frp = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_id")["frp"]
    .transform("mean")
)

# Create the deviation column.  For noise points (cluster_id == -1), this
# will be NaN because they're not in the groupby result.
df["frp_deviation"] = np.nan
df.loc[df["cluster_id"] != -1, "frp_deviation"] = (
    df.loc[df["cluster_id"] != -1, "frp"] / cluster_mean_frp
)

# Round for readability in the dashboard popups.
df["frp_deviation"] = df["frp_deviation"].round(2)

valid_devs = df["frp_deviation"].dropna()
print(f"  Computed for {len(valid_devs):,} clustered points")
print(f"  Range: {valid_devs.min():.2f}x - {valid_devs.max():.2f}x")
print(f"  Mean:  {valid_devs.mean():.2f}x   Median: {valid_devs.median():.2f}x")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 4. CREATE "alert_flag"
# ──────────────────────────────────────────────────────────────────────────────
# If a hotspot's FRP is more than 2.5x its cluster average, that's an
# unusual spike.  This could mean:
#   - An industrial accident (explosion, uncontrolled burn)
#   - A flare event at a refinery
#   - A satellite sensor anomaly (rare but possible)
#
# We flag these so the dashboard can highlight them with a distinctive border.

print("STEP 4: Setting alert_flag")

df["alert_flag"] = "Normal"
spike_mask = df["frp_deviation"] > 2.5
df.loc[spike_mask, "alert_flag"] = "Abnormal spike"

print("  alert_flag counts:")
for flag, count in df["alert_flag"].value_counts().items():
    print(f"    {flag:20s} : {count:,}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. CREATE "evidence" -- human-readable summary sentence
# ──────────────────────────────────────────────────────────────────────────────
# Each row gets a one-sentence explanation of WHY it was classified the
# way it was.  This appears in the map popup when you click a marker.
# It combines whichever pieces of evidence are relevant for that row.
#
# Example outputs:
#   "Inside OSM industrial zone; cluster active 98 days; FRP 3.26 MW
#    (0.47x cluster avg); Normal"
#   "NASA type=0 (vegetation fire); isolated detection; FRP 15.85 MW; Normal"

print("STEP 5: Building evidence strings")


def build_evidence(row):
    """Construct a one-sentence evidence string from available data."""
    parts = []

    # 1. OSM proximity info (based on the original label from classify_hotspots)
    if row["label"] == "industrial_direct":
        parts.append("Inside OSM industrial zone")
    elif row["label"] == "industrial_nearby":
        parts.append("Within 1 km of OSM industrial zone")

    # 2. Persistence info
    if row["cluster_id"] != -1 and row["distinct_days"] > 0:
        parts.append(f"cluster active {row['distinct_days']} days")
    else:
        parts.append("isolated detection (no persistent cluster)")

    # 3. FRP vs cluster average
    if pd.notna(row["frp_deviation"]):
        parts.append(
            f"FRP {row['frp']:.1f} MW ({row['frp_deviation']:.2f}x cluster avg)"
        )
    else:
        parts.append(f"FRP {row['frp']:.1f} MW")

    # 4. Alert status
    parts.append(row["alert_flag"])

    return "; ".join(parts)


df["evidence"] = df.apply(build_evidence, axis=1)

# Show a few examples so you can see what the evidence strings look like.
print("  Sample evidence strings:")
for i, sample in df.sample(3, random_state=42).iterrows():
    print(f"    [{sample['final_label']}] {sample['evidence']}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 6. EXPORT AS GeoJSON
# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON is a standard format for geospatial data that web maps (Leaflet,
# Mapbox, etc.) can load directly.  It's just JSON with a specific structure:
#
#   { "type": "FeatureCollection",
#     "features": [
#       { "type": "Feature",
#         "geometry": { "type": "Point", "coordinates": [lon, lat] },
#         "properties": { "final_label": "Industrial", "frp": 3.26, ... }
#       }, ...
#     ]
#   }
#
# We first convert our DataFrame back into a GeoDataFrame (adding a geometry
# column), then use geopandas' built-in .to_file() to write the GeoJSON.

print("STEP 6: Exporting GeoJSON")

# Build geometry from lat/lon columns.
geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Format acq_date as a string so it serializes cleanly into JSON.
gdf["acq_date"] = pd.to_datetime(gdf["acq_date"]).dt.strftime("%Y-%m-%d")

# Select only the columns the dashboard needs.  Keeping the GeoJSON lean
# makes the file smaller and the dashboard faster.
export_cols = [
    "geometry",        # Point(lon, lat) -- becomes "coordinates" in GeoJSON
    "final_label",     # "Industrial" or "Natural Fire"
    "frp",             # Fire Radiative Power in MW
    "frp_deviation",   # ratio vs cluster mean (NaN -> null in JSON)
    "alert_flag",      # "Normal" or "Abnormal spike"
    "evidence",        # human-readable summary sentence
    "acq_date",        # acquisition date string
    "distinct_days",   # how many days the cluster was active
]

gdf_export = gdf[export_cols]

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "data", "firms_final.geojson")
gdf_export.to_file(OUTPUT_PATH, driver="GeoJSON")

file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"  Saved to: {OUTPUT_PATH}")
print(f"  Features: {len(gdf_export):,}   File size: {file_size_kb:.0f} KB")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 7. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("FINAL SUMMARY")
print("=" * 72)
print(f"\n  Total hotspots exported: {len(gdf_export):,}")

print("\n  final_label breakdown:")
for label, count in df["final_label"].value_counts().items():
    pct = 100 * count / len(df)
    print(f"    {label:20s} : {count:,}  ({pct:.1f}%)")

print("\n  alert_flag breakdown:")
for flag, count in df["alert_flag"].value_counts().items():
    pct = 100 * count / len(df)
    print(f"    {flag:20s} : {count:,}  ({pct:.1f}%)")

print("\nDone! ->  Open dashboard.html to see the map.")
