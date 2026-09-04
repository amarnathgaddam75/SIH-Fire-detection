"""
classify_hotspots.py  —  FIRMS Fire-vs-Industrial Hotspot Classifier
====================================================================
Loads NASA FIRMS CSV data, enriches it with OpenStreetMap industrial-zone
information, clusters persistent thermal sources, and labels each hotspot
as industrial or natural fire.

Dependencies: pandas, geopandas, osmnx, scikit-learn, shapely
Install:      pip install pandas geopandas osmnx scikit-learn shapely

Author:  SIH Hackathon Prototype
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
# In C you'd #include headers.  In Python we 'import' modules.
# Each module here serves a specific purpose in the pipeline.

import os
import warnings

import numpy as np                      # numerical arrays  (like C arrays, but safer)
import pandas as pd                     # tabular data  (think: spreadsheet in memory)
import geopandas as gpd                 # pandas + geometry  (adds spatial ops)
import osmnx as ox                      # queries OpenStreetMap data via Overpass API
from shapely.geometry import Point      # represents a lat/lon coordinate as an object
from sklearn.cluster import DBSCAN      # density-based clustering  (groups nearby points)

# Suppress some noisy warnings from libraries so our output stays readable.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE THE TWO FIRMS CSV FILES
# ──────────────────────────────────────────────────────────────────────────────
# NASA distributes FIRMS data in two files:
#   - "archive"  → verified historical data, includes a 'type' column
#                   (type=0 = presumed vegetation fire, type=2 = static source)
#   - "nrt"      → near-real-time data, same schema MINUS the 'type' column
#
# We load both, add the missing 'type' column to nrt (filled with NaN, meaning
# "unknown"), then stack them into one DataFrame sorted by date.

# Build paths relative to this script's location so it works from any cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data", "DL_FIRE_J1V-C2_795760")

archive_path = os.path.join(DATA_DIR, "fire_archive_J1V-C2_795760.csv")
nrt_path     = os.path.join(DATA_DIR, "fire_nrt_J1V-C2_795760.csv")

print("=" * 72)
print("STEP 1: Loading FIRMS CSV data")
print("=" * 72)

# pd.read_csv() is like fopen+fscanf but does all the parsing for you.
df_archive = pd.read_csv(archive_path)
df_nrt     = pd.read_csv(nrt_path)

print(f"  Archive rows : {len(df_archive):,}   columns: {list(df_archive.columns)}")
print(f"  NRT rows     : {len(df_nrt):,}   columns: {list(df_nrt.columns)}")

# The NRT file has no 'type' column.  Add it filled with NaN so both DataFrames
# have the same schema and can be concatenated cleanly.
# NaN (Not a Number) is Python/pandas's way of saying "missing value" — like
# a NULL pointer but for data.
df_nrt["type"] = np.nan

# pd.concat stacks DataFrames vertically (like appending one array to another).
# ignore_index=True re-numbers the rows 0..N instead of keeping the originals.
df = pd.concat([df_archive, df_nrt], ignore_index=True)

# Convert the date string "2026-03-01" into a proper date object so sorting
# and date-based grouping work correctly (string sort would also work for
# ISO dates, but this is more explicit and lets us do date arithmetic later).
df["acq_date"] = pd.to_datetime(df["acq_date"])
df = df.sort_values("acq_date").reset_index(drop=True)

print(f"  Combined     : {len(df):,} rows, date range "
      f"{df['acq_date'].min().date()} → {df['acq_date'].max().date()}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONVERT TO A GeoDataFrame  (add spatial geometry)
# ──────────────────────────────────────────────────────────────────────────────
# A GeoDataFrame is just a DataFrame where one column holds geometry objects
# (Point, Polygon, etc.).  This lets us do spatial queries like "is this point
# inside that polygon?" — impossible with plain CSV numbers.
#
# CRS = Coordinate Reference System.  EPSG:4326 means "latitude/longitude in
# degrees on the WGS84 ellipsoid" — the GPS standard.  We set it here so that
# geopandas knows what units our coordinates are in.

print("STEP 2: Building GeoDataFrame")

# Create a Point object for each row from its lat/lon.
# Note: Point takes (x, y) = (longitude, latitude) — the GIS convention is
# the opposite of the geographic convention (lat first).  This is a common
# gotcha; getting it wrong silently puts your points in the wrong hemisphere.
geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]

gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
print(f"  GeoDataFrame with {len(gdf)} points, CRS = {gdf.crs}")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 3. QUERY OpenStreetMap FOR INDUSTRIAL INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────────────
# OpenStreetMap (OSM) is a crowd-sourced map of the world.  The Overpass API
# lets you query it by bounding box + tags.  We ask for features tagged as
# industrial land, power plants, manufacturing works, or mines.
#
# osmnx wraps the Overpass API so we don't have to write raw Overpass QL.
# It returns a GeoDataFrame of polygons (building footprints, land-use zones).
#
# Bounding box for the Korba-Raigarh industrial belt, Chhattisgarh:
#   North 22.6, South 22.1, East 83.1, West 82.3

print("STEP 3: Querying OSM for industrial features (this may take a moment)...")

BBOX_NORTH = 22.6
BBOX_SOUTH = 22.1
BBOX_EAST  = 83.1
BBOX_WEST  = 82.3

# We query multiple tag sets because different mappers use different tags for
# the same kind of industrial feature.  Each dict is one tag filter.
#   landuse=industrial    → zones designated for factories/industry
#   power=plant           → electricity generation stations
#   man_made=works        → manufacturing/processing plants
#   landuse=quarry        → open-pit mines
#   man_made=mineshaft    → underground mine entrances
osm_tags = {
    "landuse": ["industrial", "quarry"],
    "power": "plant",
    "man_made": ["works", "mineshaft"],
}

# ox.features_from_bbox returns all OSM features in the bounding box matching
# the given tags.  The result is a GeoDataFrame with a mix of geometry types
# (points, lines, polygons) depending on how the feature was mapped.
try:
    # osmnx 2.1 bbox order: (west, south, east, north) = (left, bottom, right, top)
    osm_gdf = ox.features_from_bbox(
        bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
        tags=osm_tags,
    )
    print(f"  Retrieved {len(osm_gdf)} OSM features")

    # Keep only polygon/multipolygon geometries — we need areas to check if
    # a fire point falls "inside" an industrial zone.  Point features from OSM
    # (e.g., a single tagged node) can't be used for containment checks.
    osm_polygons = osm_gdf[
        osm_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    print(f"  Of which {len(osm_polygons)} are polygons (usable for spatial join)")

except Exception as e:
    # If the Overpass API is down or rate-limited, don't crash the whole script.
    # Instead, create an empty GeoDataFrame so subsequent steps still run
    # (they'll just label everything as "unclassified").
    print(f"  ⚠ OSM query failed: {e}")
    print("  Continuing with empty industrial polygons (all points will be 'unclassified')")
    osm_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

print()


# ──────────────────────────────────────────────────────────────────────────────
# 4. SPATIAL JOIN — label hotspots by proximity to industrial zones
# ──────────────────────────────────────────────────────────────────────────────
# This is the core classification logic.  For each fire hotspot, we ask:
#   A) Is it INSIDE an OSM industrial polygon?         → "industrial_direct"
#   B) Is it within 1 km of one but not inside?        → "industrial_nearby"
#   C) Neither?                                        → "unclassified"
#
# IMPORTANT CRS note:
#   EPSG:4326 uses degrees.  Buffering by 0.01° ≈ ~1.1 km at the equator, but
#   the actual distance varies with latitude and is different in x vs y.
#   For accurate metric buffers we MUST reproject to a *projected* CRS that
#   uses metres.  EPSG:32644 = UTM zone 44N, which is correct for longitude
#   ~78°–84° E in India.  After buffering, we reproject back to 4326 for the
#   spatial join against our 4326 hotspot points.

print("STEP 4: Spatial join — classifying hotspots")

# Start with a default label.  We'll overwrite it for points that match.
gdf["label"] = "unclassified"

if len(osm_polygons) > 0:
    # ── 4a. Direct containment join ──
    # gpd.sjoin with predicate="within" checks: is the point inside any polygon?
    # 'how="inner"' means: only keep rows where a match was found.
    # The result has an 'index_right' column telling us WHICH polygon matched.
    joined_direct = gpd.sjoin(gdf, osm_polygons, how="inner", predicate="within")

    # Mark those hotspot indices as "industrial_direct".
    # A hotspot might be inside MULTIPLE overlapping polygons — we just need to
    # know it's inside at least one, so we take unique indices.
    direct_indices = joined_direct.index.unique()
    gdf.loc[direct_indices, "label"] = "industrial_direct"
    print(f"  industrial_direct : {len(direct_indices):,} hotspots inside a polygon")

    # ── 4b. Nearby (within 1 km buffer) ──
    # Reproject polygons to UTM so we can buffer in metres, not degrees.
    osm_utm = osm_polygons.to_crs("EPSG:32644")

    # Buffer each polygon by 1000 metres (1 km).  This creates a new, larger
    # polygon that extends 1 km outward from the original boundary.
    osm_buffered = osm_utm.copy()
    osm_buffered["geometry"] = osm_utm.geometry.buffer(1000)  # 1000 m = 1 km

    # Reproject the buffered polygons back to 4326 to match the hotspot CRS.
    osm_buffered = osm_buffered.to_crs("EPSG:4326")

    # Spatial join again, this time against the buffered polygons.
    joined_nearby = gpd.sjoin(gdf, osm_buffered, how="inner", predicate="within")
    nearby_indices = joined_nearby.index.unique()

    # Only label points that are NEARBY but NOT DIRECTLY INSIDE (avoid
    # downgrading the "industrial_direct" label).
    nearby_only = nearby_indices.difference(direct_indices)
    gdf.loc[nearby_only, "label"] = "industrial_nearby"
    print(f"  industrial_nearby : {len(nearby_only):,} hotspots within 1 km buffer")

else:
    print("  No OSM polygons available — skipping spatial join")

unclassified_count = (gdf["label"] == "unclassified").sum()
print(f"  unclassified      : {unclassified_count:,} hotspots")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. PERSISTENCE CLUSTERING  (find locations with repeated detections)
# ──────────────────────────────────────────────────────────────────────────────
# A forest fire burns for a few days then moves on.  A coal plant or smelter
# emits heat every single day, forever.  By clustering hotspots spatially and
# counting how many *different* days each cluster was detected, we can
# identify "persistent thermal sources" without any map data at all.
#
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
#   - Groups points that are within `eps` distance of each other
#   - `eps` = 500 m ≈ reasonable size for an industrial facility
#   - `min_samples=2` = a cluster needs at least 2 detections (not just one)
#   - Points that don't belong to any cluster get label -1 ("noise")
#
# We work in UTM (metres) so that eps=500 actually means 500 metres.

print("STEP 5: Persistence clustering (DBSCAN, eps=500 m)")

# Reproject all hotspot points to UTM for accurate distance calculations.
gdf_utm = gdf.to_crs("EPSG:32644")

# Extract x,y coordinates as a 2D numpy array — DBSCAN needs a plain matrix,
# not geometry objects.  This is like extracting raw doubles from structs in C.
coords = np.column_stack([gdf_utm.geometry.x, gdf_utm.geometry.y])

# Run DBSCAN.  eps=500 means "two points are neighbors if ≤ 500 m apart".
clustering = DBSCAN(eps=500, min_samples=2).fit(coords)

# The .labels_ attribute is an array of cluster IDs, one per point.
# -1 means "noise" (point doesn't belong to any cluster).
gdf["cluster_id"] = clustering.labels_

n_clusters = len(set(clustering.labels_) - {-1})
n_noise    = (clustering.labels_ == -1).sum()
print(f"  Found {n_clusters} spatial clusters + {n_noise:,} noise points")

# For each cluster, count how many DISTINCT dates had at least one detection.
# A cluster with detections on 50 different days is almost certainly a
# permanent industrial heat source, not a wildfire.
#
# We group by cluster_id, then count unique acq_date values.
cluster_day_counts = (
    gdf[gdf["cluster_id"] != -1]           # exclude noise points
    .groupby("cluster_id")["acq_date"]     # group by cluster
    .nunique()                              # count distinct dates
    .rename("distinct_days")               # give the series a name
)

# Map that count back to every point in the cluster.
# Points in cluster 7 all get the same "distinct_days" value.
gdf["distinct_days"] = gdf["cluster_id"].map(cluster_day_counts).fillna(0).astype(int)

# Show the top clusters by persistence.
print("\n  Top persistent clusters:")
top_clusters = (
    gdf[gdf["cluster_id"] != -1]
    .groupby("cluster_id")
    .agg(
        n_detections=("cluster_id", "size"),
        distinct_days=("distinct_days", "first"),
        mean_lat=("latitude", "mean"),
        mean_lon=("longitude", "mean"),
        mean_frp=("frp", "mean"),
    )
    .sort_values("distinct_days", ascending=False)
    .head(10)
)
print(top_clusters.to_string(index=True))
print()


# ──────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY — cross-reference labels, NASA type, and persistence
# ──────────────────────────────────────────────────────────────────────────────
# NASA's "type" column in the archive data:
#   type=0 → presumed vegetation fire
#   type=2 → known static source (industrial, volcano, offshore)
#   NaN    → NRT data (type not yet assigned)
#
# We compare our OSM-based labels and DBSCAN persistence clusters against
# NASA's type=2 flag to see if our classification agrees with theirs.

print("=" * 72)
print("STEP 6: Summary & Validation")
print("=" * 72)

print(f"\n  Total hotspots: {len(gdf):,}")

print("\n  Classification label counts:")
for label, count in gdf["label"].value_counts().items():
    print(f"    {label:25s} : {count:,}")

print(f"\n  NASA type column distribution:")
for t, count in gdf["type"].value_counts(dropna=False).sort_index().items():
    label = {0: "vegetation fire", 2: "static source"}.get(t, "NRT (unknown)")
    print(f"    type={str(t):5s} ({label:17s}) : {count:,}")

# Cross-tabulation: how do our labels compare to NASA's type?
print("\n  Cross-tab: our label vs NASA type:")
ct = pd.crosstab(gdf["label"], gdf["type"].fillna(-1).astype(int), margins=True)
ct.columns = [f"type={c}" if c != "All" else "All" for c in ct.columns]
print(ct.to_string())

# Overlap check: of the type=2 (static source) rows, how many did our
# persistence clustering also flag as persistent (≥ 10 distinct days)?
type2_rows = gdf[gdf["type"] == 2]
persistent_mask = type2_rows["distinct_days"] >= 10
print(f"\n  Validation — NASA type=2 rows: {len(type2_rows)}")
print(f"    Also in a persistent cluster (≥10 days): "
      f"{persistent_mask.sum()} ({100*persistent_mask.mean():.1f}%)")
print(f"    Not in a persistent cluster:             "
      f"{(~persistent_mask).sum()} ({100*(~persistent_mask).mean():.1f}%)")

# And the reverse: of our persistent clusters, how many contain type=2 points?
persistent_points = gdf[(gdf["cluster_id"] != -1) & (gdf["distinct_days"] >= 10)]
has_type2 = persistent_points["type"] == 2
print(f"\n  Reverse check — our persistent points (≥10 days): {len(persistent_points)}")
print(f"    Have NASA type=2: {has_type2.sum()} ({100*has_type2.mean():.1f}%)")
print()


# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE THE FINAL LABELED RESULT
# ──────────────────────────────────────────────────────────────────────────────
# Save as a plain CSV (drop the geometry column since CSV can't store it;
# the latitude/longitude columns already have the coordinate info).

output_path = os.path.join(SCRIPT_DIR, "data", "firms_labeled.csv")

# Drop the geometry column — it's not serializable to CSV and the lat/lon
# columns already contain the same information in a CSV-friendly format.
out_df = gdf.drop(columns=["geometry"])
out_df.to_csv(output_path, index=False)

print(f"STEP 7: Saved labeled data to {output_path}")
print(f"        Shape: {out_df.shape[0]} rows × {out_df.shape[1]} columns")
print(f"        Columns: {list(out_df.columns)}")
print("\nDone! ✓")
