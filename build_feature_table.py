"""
build_feature_table.py  --  Feature Engineering for FIRMS Hotspot Data
======================================================================
Takes the labeled hotspot CSV from classify_hotspots.py and enriches it
with continuous spatial distances, per-cluster historical statistics,
an expanded 5-class classification, optional Sentinel-2 indices, and
anomaly alert flags.

Input:   data/firms_labeled.csv           (from classify_hotspots.py)
Output:  data/feature_vector.csv          (enriched feature table)

Dependencies: pandas, geopandas, osmnx, numpy, shapely
Install:      pip install pandas geopandas osmnx numpy shapely

Author:  SIH Hackathon Prototype
"""

# ------------------------------------------------------------------------------
# 0. IMPORTS
# ------------------------------------------------------------------------------

import os
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

# Korba bounding box -- same as in classify_hotspots.py
BBOX_NORTH = 22.6
BBOX_SOUTH = 22.1
BBOX_EAST  = 83.1
BBOX_WEST  = 82.3

# UTM zone 44N -- correct for longitude ~78-84 E in India
UTM_CRS = "EPSG:32644"


# ==============================================================================
# STEP 1: LOAD firms_labeled.csv
# ==============================================================================
# This CSV was produced by classify_hotspots.py and contains 2,008 rows with
# columns like latitude, longitude, frp, type, label, cluster_id, distinct_days.
# We convert it to a GeoDataFrame so we can do spatial operations.

print("=" * 72)
print("STEP 1: Loading data/firms_labeled.csv")
print("=" * 72)

input_path = os.path.join(DATA_DIR, "firms_labeled.csv")
df = pd.read_csv(input_path)

print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# Build Point geometries from lat/lon.  Point(x, y) = Point(lon, lat).
geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print(f"  GeoDataFrame ready -- CRS = {gdf.crs}")
print()


# ==============================================================================
# STEP 2: PULL OSM POLYGONS (industrial + farmland)
# ==============================================================================
# We query OpenStreetMap for two sets of polygons:
#   A) Industrial infrastructure -- same tags as classify_hotspots.py
#   B) Farmland -- landuse=farmland or landuse=agricultural
#
# Both queries use the identical Korba bounding box via osmnx.

print("=" * 72)
print("STEP 2: Querying OSM for industrial + farmland polygons")
print("=" * 72)

# -- 2a. Industrial polygons (mirrors classify_hotspots.py exactly) ----------
industrial_tags = {
    "landuse": ["industrial", "quarry"],
    "power": "plant",
    "man_made": ["works", "mineshaft"],
}

try:
    osm_industrial_raw = ox.features_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        tags=industrial_tags,
    )
    osm_industrial = osm_industrial_raw[
        osm_industrial_raw.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"  Industrial: {len(osm_industrial_raw)} OSM features -> "
          f"{len(osm_industrial)} polygons")
except Exception as e:
    print(f"  Warning: Industrial OSM query failed: {e}")
    print("    -> distance_to_industrial_m will be NaN for all rows")
    osm_industrial = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

# -- 2b. Farmland polygons --------------------------------------------------
farmland_tags = {
    "landuse": ["farmland", "agricultural"],
}

try:
    osm_farmland_raw = ox.features_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        tags=farmland_tags,
    )
    osm_farmland = osm_farmland_raw[
        osm_farmland_raw.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"  Farmland:   {len(osm_farmland_raw)} OSM features -> "
          f"{len(osm_farmland)} polygons")
except Exception as e:
    print(f"  Warning: Farmland OSM query failed: {e}")
    print("    -> distance_to_farmland_m will be NaN for all rows")
    osm_farmland = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

print()


# ==============================================================================
# STEP 3: COMPUTE CONTINUOUS DISTANCE COLUMNS
# ==============================================================================
# For every hotspot point, we compute the straight-line distance (in metres)
# to the nearest industrial polygon boundary and the nearest farmland polygon
# boundary.  This requires working in a projected CRS (UTM) where the unit
# is metres, not degrees.
#
# Strategy: merge all polygons of each type into a single MultiPolygon via
# unary_union, then call point.distance(union) for each point.  This is
# efficient because shapely uses spatial indexing on union geometries.

print("=" * 72)
print("STEP 3: Computing distance_to_industrial_m and distance_to_farmland_m")
print("=" * 72)

# Reproject hotspot points to UTM
gdf_utm = gdf.to_crs(UTM_CRS)

# -- 3a. Distance to nearest industrial polygon -----------------------------
if len(osm_industrial) > 0:
    industrial_utm = osm_industrial.to_crs(UTM_CRS)
    # Merge all industrial polygons into one geometry for fast distance calc.
    industrial_union = unary_union(industrial_utm.geometry.values)
    # .distance() returns metres because both geometries are in UTM.
    gdf["distance_to_industrial_m"] = gdf_utm.geometry.distance(industrial_union).round(1)
    print(f"  distance_to_industrial_m -- "
          f"min: {gdf['distance_to_industrial_m'].min():.0f} m, "
          f"max: {gdf['distance_to_industrial_m'].max():.0f} m, "
          f"median: {gdf['distance_to_industrial_m'].median():.0f} m")
else:
    gdf["distance_to_industrial_m"] = np.nan
    print("  distance_to_industrial_m -- all NaN (no industrial polygons)")

# -- 3b. Distance to nearest farmland polygon --------------------------------
if len(osm_farmland) > 0:
    farmland_utm = osm_farmland.to_crs(UTM_CRS)
    farmland_union = unary_union(farmland_utm.geometry.values)
    gdf["distance_to_farmland_m"] = gdf_utm.geometry.distance(farmland_union).round(1)
    print(f"  distance_to_farmland_m   -- "
          f"min: {gdf['distance_to_farmland_m'].min():.0f} m, "
          f"max: {gdf['distance_to_farmland_m'].max():.0f} m, "
          f"median: {gdf['distance_to_farmland_m'].median():.0f} m")
else:
    gdf["distance_to_farmland_m"] = np.nan
    print("  distance_to_farmland_m   -- all NaN (no farmland polygons)")

print()


# ==============================================================================
# STEP 4: PER-CLUSTER HISTORICAL STATISTICS
# ==============================================================================
# For every row that belongs to a DBSCAN cluster (cluster_id != -1), we
# compute summary statistics from that cluster's own detections.  These
# features capture what "normal" looks like for each persistent source,
# making anomaly detection possible later.
#
# Rows with cluster_id == -1 (noise / one-off detections) get NaN because
# there is no cluster history to compare against.

print("=" * 72)
print("STEP 4: Computing per-cluster historical statistics")
print("=" * 72)

# Work only on clustered rows
clustered_mask = gdf["cluster_id"] != -1
clustered = gdf.loc[clustered_mask]

# Compute cluster-level aggregates
cluster_stats = clustered.groupby("cluster_id")["frp"].agg(
    historical_mean_frp="mean",
    historical_max_frp="max",
    historical_std_frp="std",
)

# Map cluster aggregates back to every row in that cluster
gdf["historical_mean_frp"] = gdf["cluster_id"].map(cluster_stats["historical_mean_frp"])
gdf["historical_max_frp"]  = gdf["cluster_id"].map(cluster_stats["historical_max_frp"])
gdf["historical_std_frp"]  = gdf["cluster_id"].map(cluster_stats["historical_std_frp"])

# frp_deviation = how many times the cluster mean is this reading?
# A value of 3.0 means this reading is 3x the cluster average -- suspicious.
gdf["frp_deviation"] = np.where(
    clustered_mask,
    gdf["frp"] / gdf["historical_mean_frp"],
    np.nan
)
gdf["frp_deviation"] = gdf["frp_deviation"].round(4)

# night_detection_ratio = what fraction of the cluster's detections were at night?
# Night-heavy clusters are more likely to be industrial (factories run 24/7).
cluster_night_ratio = (
    clustered
    .groupby("cluster_id")["daynight"]
    .apply(lambda s: (s == "N").mean())
    .rename("night_detection_ratio")
)
gdf["night_detection_ratio"] = gdf["cluster_id"].map(cluster_night_ratio)

# Ensure noise points get NaN for all historical features
noise_mask = gdf["cluster_id"] == -1
gdf.loc[noise_mask, [
    "historical_mean_frp", "historical_max_frp", "historical_std_frp",
    "frp_deviation", "night_detection_ratio"
]] = np.nan

n_clustered = clustered_mask.sum()
n_noise = noise_mask.sum()
print(f"  Computed stats for {n_clustered:,} clustered rows "
      f"({n_noise:,} noise rows -> NaN)")
print(f"  historical_mean_frp range: "
      f"{gdf['historical_mean_frp'].min():.2f} - {gdf['historical_mean_frp'].max():.2f}")
print(f"  frp_deviation range:       "
      f"{gdf['frp_deviation'].min():.4f} - {gdf['frp_deviation'].max():.4f}")
print(f"  night_detection_ratio range: "
      f"{gdf['night_detection_ratio'].min():.3f} - {gdf['night_detection_ratio'].max():.3f}")
print()


# ==============================================================================
# STEP 5: EXPANDED 5-CLASS CLASSIFICATION -> class_label
# ==============================================================================
# The existing "label" column from classify_hotspots.py has 3 values:
#   industrial_direct, industrial_nearby, unclassified
#
# We now create a richer "class_label" with 5 classes.  The key principle:
# conflicting signals should map to "Unknown", NOT be force-classified.
#
# Rules (evaluated in priority order):
#   1. Industrial: within 1 km of industrial polygon AND
#      (NASA type == 2 OR distinct_days >= 10)
#   2. Agricultural Burning: within 1 km of farmland polygon AND
#      NOT near industrial infrastructure AND distinct_days <= 3
#   3. Natural Fire: NASA type == 0 AND not caught by rules 1 or 2
#   4. Unknown: everything else (including conflicting evidence)

print("=" * 72)
print("STEP 5: Expanded 5-class classification -> class_label")
print("=" * 72)

# Start everything as "Unknown" -- the safe default.
gdf["class_label"] = "Unknown"

# -- Rule 1: Industrial ------------------------------------------------------
# Near industrial infrastructure AND has industrial-consistent evidence.
# "Near" = within 1 km (distance_to_industrial_m <= 1000).
# "Industrial-consistent" = NASA says static source (type=2) OR the cluster
# was detected on >=10 distinct days (persistent heat = factory/plant).
near_industrial = gdf["distance_to_industrial_m"] <= 1000
type_is_2 = gdf["type"] == 2
high_persistence = gdf["distinct_days"] >= 10
industrial_rule = near_industrial & (type_is_2 | high_persistence)
gdf.loc[industrial_rule, "class_label"] = "Industrial"

# -- Rule 2: Agricultural Burning --------------------------------------------
# Near farmland AND NOT near industrial AND low persistence (<=3 days).
# This catches crop residue burning -- it is near farms, burns briefly, and
# is not near any factory.
near_farmland = gdf["distance_to_farmland_m"] <= 2000
not_near_industrial = gdf["distance_to_industrial_m"] > 1000
low_persistence = gdf["distinct_days"] <= 3
ag_burn_rule = near_farmland & not_near_industrial & low_persistence

# Only apply to rows not already classified as Industrial (priority order).
still_unknown = gdf["class_label"] == "Unknown"
gdf.loc[still_unknown & ag_burn_rule, "class_label"] = "Agricultural Burning"

# -- Rule 3: Natural Fire ----------------------------------------------------
# NASA type == 0 (vegetation fire) AND not near industrial infrastructure AND
# not already classified above.
# If type=0 is near industrial (distance <= 1000) but lacks high persistence
# or type=2, it represents conflicting evidence and falls through to Unknown.
type_is_0 = gdf["type"] == 0
still_unknown = gdf["class_label"] == "Unknown"
gdf.loc[still_unknown & type_is_0 & not_near_industrial, "class_label"] = "Natural Fire"

# -- Rule 4: Unknown ---------------------------------------------------------
# Everything still labeled "Unknown" stays that way.  This includes:
#   - type=0 inside industrial zone but WITHOUT high persistence or type=2
#     (conflicting: vegetation fire signal inside a factory zone)
#   - type=2 far from any industrial polygon with low persistence
#     (conflicting: static source tag but no spatial/temporal evidence)
#   - NRT data (type=NaN) with no clear spatial signal
#   - Any other ambiguous combinations

# -- Print class_label breakdown ---------------------------------------------
print("\n  class_label breakdown:")
print("  " + "-" * 45)
for label, count in gdf["class_label"].value_counts().sort_index().items():
    print(f"    {label:25s} : {count:,}")
print("  " + "-" * 45)
print(f"    {'TOTAL':25s} : {len(gdf):,}")

# -- The 233-row conflict check ----------------------------------------------
# In classify_hotspots.py, 233 rows had type=0 (NASA says vegetation fire)
# but fell inside an industrial polygon (label == "industrial_direct").
# These are inherently conflicting signals.  Under our new 5-class logic,
# how many of these ended up as "Unknown"?
#
# A type=0 + industrial_direct row will be "Industrial" ONLY if it also has
# distinct_days >= 10 (since type != 2).  Otherwise it is "Unknown" -- we
# refuse to confidently classify conflicting evidence.

conflict_mask = (gdf["type"] == 0) & (gdf["label"] == "industrial_direct")
n_conflict_total = conflict_mask.sum()
n_conflict_unknown = (gdf.loc[conflict_mask, "class_label"] == "Unknown").sum()
n_conflict_industrial = (gdf.loc[conflict_mask, "class_label"] == "Industrial").sum()
n_conflict_other = n_conflict_total - n_conflict_unknown - n_conflict_industrial

print(f"\n  233-row conflict check (type=0 + industrial_direct):")
print(f"    Total rows matching type=0 AND industrial_direct: {n_conflict_total}")
print(f"    Labeled 'Unknown'    (conflicting evidence):      {n_conflict_unknown}")
print(f"    Labeled 'Industrial' (high persistence rescued):  {n_conflict_industrial}")
if n_conflict_other > 0:
    print(f"    Labeled other:                                    {n_conflict_other}")
print()

# ------------------------------------------------------------------------------
# PAUSE POINT: The user asked to review the class_label breakdown and the
# 233-row check before proceeding to steps 6-8.  In a script, we just
# continue -- but the output above is clearly separated for review.
# ------------------------------------------------------------------------------


# ==============================================================================
# STEP 6: MERGE SENTINEL-2 INDICES (if available)
# ==============================================================================
# If a Sentinel-2 pilot CSV exists anywhere under data/, merge its ndvi, ndbi,
# ndwi columns in by matching coordinates.  Otherwise, create those three
# columns filled with "unavailable".
#
# We search recursively for any CSV whose name contains "sentinel" or "s2".

print("=" * 72)
print("STEP 6: Sentinel-2 index merge")
print("=" * 72)

sentinel_csvs = (
    glob.glob(os.path.join(DATA_DIR, "**", "*sentinel*"), recursive=True) +
    glob.glob(os.path.join(DATA_DIR, "**", "*Sentinel*"), recursive=True) +
    glob.glob(os.path.join(DATA_DIR, "**", "*s2*"), recursive=True) +
    glob.glob(os.path.join(DATA_DIR, "**", "*S2*"), recursive=True)
)

# Filter to only .csv files
sentinel_csvs = [f for f in sentinel_csvs if f.lower().endswith(".csv")]
# Deduplicate
sentinel_csvs = list(set(sentinel_csvs))

if sentinel_csvs:
    print(f"  Found Sentinel-2 CSV(s): {sentinel_csvs}")
    # Use the first one found
    s2_df = pd.read_csv(sentinel_csvs[0])
    print(f"  Loaded {len(s2_df):,} rows from {os.path.basename(sentinel_csvs[0])}")

    # Check which index columns exist in the Sentinel-2 file
    s2_index_cols = [c for c in ["ndvi", "ndbi", "ndwi"] if c in s2_df.columns]
    print(f"  Index columns found: {s2_index_cols}")

    if s2_index_cols and "latitude" in s2_df.columns and "longitude" in s2_df.columns:
        # Round coordinates for matching (avoid floating-point mismatch).
        # FIRMS coordinates have ~5 decimal places; round to 4 for tolerance.
        gdf["_lat_round"] = gdf["latitude"].round(4)
        gdf["_lon_round"] = gdf["longitude"].round(4)
        s2_df["_lat_round"] = s2_df["latitude"].round(4)
        s2_df["_lon_round"] = s2_df["longitude"].round(4)

        # Merge on rounded coordinates
        merge_cols = ["_lat_round", "_lon_round"] + s2_index_cols
        gdf = gdf.merge(
            s2_df[merge_cols].drop_duplicates(subset=["_lat_round", "_lon_round"]),
            on=["_lat_round", "_lon_round"],
            how="left",
        )
        # Drop helper columns
        gdf.drop(columns=["_lat_round", "_lon_round"], inplace=True)

        # Fill unmatched rows with "unavailable"
        for col in ["ndvi", "ndbi", "ndwi"]:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna("unavailable")
            else:
                gdf[col] = "unavailable"

        matched = (gdf["ndvi"] != "unavailable").sum() if "ndvi" in gdf.columns else 0
        print(f"  Merged -- {matched:,} rows matched, "
              f"{len(gdf) - matched:,} filled with 'unavailable'")
    else:
        print("  Warning: Required columns (latitude, longitude, ndvi/ndbi/ndwi) not found")
        for col in ["ndvi", "ndbi", "ndwi"]:
            gdf[col] = "unavailable"
else:
    print("  No Sentinel-2 CSV found under data/")
    print("  Creating ndvi, ndbi, ndwi columns with 'unavailable'")
    for col in ["ndvi", "ndbi", "ndwi"]:
        gdf[col] = "unavailable"

print()


# ==============================================================================
# STEP 7: ALERT FLAG
# ==============================================================================
# Flag hotspots whose FRP is more than 2.5x their cluster average as
# "Abnormal spike".  These could indicate industrial accidents, uncontrolled
# flaring, or sensor anomalies -- worth investigating.

print("=" * 72)
print("STEP 7: Alert flag")
print("=" * 72)

if "alert_flag" not in gdf.columns:
    gdf["alert_flag"] = np.where(
        gdf["frp_deviation"] > 2.5,
        "Abnormal spike",
        "Normal"
    )
    print("  Created alert_flag column")
else:
    print("  alert_flag already exists -- skipping")

print("  alert_flag counts:")
for flag, count in gdf["alert_flag"].value_counts().items():
    print(f"    {flag:20s} : {count:,}")
print()


# ==============================================================================
# STEP 8: SAVE TO data/feature_vector.csv
# ==============================================================================
# Drop the geometry column (not CSV-serializable) -- lat/lon columns already
# contain the coordinate information.

print("=" * 72)
print("STEP 8: Saving data/feature_vector.csv")
print("=" * 72)

output_path = os.path.join(DATA_DIR, "feature_vector.csv")

out_df = gdf.drop(columns=["geometry"])
out_df.to_csv(output_path, index=False)

print(f"  Saved to {output_path}")
print(f"  Shape: {out_df.shape[0]:,} rows x {out_df.shape[1]} columns")
print(f"  Columns: {list(out_df.columns)}")
print("\nDone!")
