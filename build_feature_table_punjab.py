"""
build_feature_table_punjab.py
=============================
Feature Engineering & 5-Class Labeling for Punjab Stubble-Burning Belt.

Input:   data_punjab/firms_labeled_punjab.csv
Output:  data_punjab/feature_vector_punjab.csv

Bounding Box: North 30.6, South 29.6, East 76.5, West 75.0
Projected CRS: EPSG:32643 (UTM Zone 43N)
"""

import os
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
DATA_DIR = os.path.join(SCRIPT_DIR, "data_punjab")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BBOX_NORTH = 30.6
BBOX_SOUTH = 29.6
BBOX_EAST  = 76.5
BBOX_WEST  = 75.0
UTM_CRS    = "EPSG:32643"

INPUT_CSV = os.path.join(DATA_DIR, "firms_labeled_punjab.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "feature_vector_punjab.csv")

CACHE_IND_PATH = os.path.join(CACHE_DIR, "osm_industrial_polygons_punjab.geojson")
CACHE_FARM_PATH = os.path.join(CACHE_DIR, "osm_farmland_polygons_punjab.geojson")


def main():
    print("=" * 72)
    print("STEP 1: Loading data_punjab/firms_labeled_punjab.csv")
    print("=" * 72)

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf_utm = gdf.to_crs(UTM_CRS)

    # Step 2: OSM Polygons (Industrial & Farmland)
    print("\n" + "=" * 72)
    print("STEP 2: Loading / Querying OSM industrial and farmland polygons")
    print("=" * 72)

    # Industrial
    if os.path.exists(CACHE_IND_PATH):
        print(f"  Loading industrial polygons from cache: {CACHE_IND_PATH}")
        osm_industrial = gpd.read_file(CACHE_IND_PATH)
    else:
        industrial_tags = {
            "landuse": ["industrial", "quarry"],
            "power": "plant",
            "man_made": ["works", "mineshaft"],
        }
        raw_ind = ox.features_from_bbox(
            bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
            tags=industrial_tags,
        )
        osm_industrial = raw_ind[raw_ind.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        osm_industrial.to_file(CACHE_IND_PATH, driver="GeoJSON")

    print(f"  Industrial polygons: {len(osm_industrial)}")

    # Farmland
    if os.path.exists(CACHE_FARM_PATH):
        print(f"  Loading farmland polygons from cache: {CACHE_FARM_PATH}")
        osm_farmland = gpd.read_file(CACHE_FARM_PATH)
    else:
        print(f"  Querying Overpass for farmland in bbox ({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH})...")
        farmland_tags = {"landuse": ["farmland", "agricultural"]}
        raw_farm = ox.features_from_bbox(
            bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
            tags=farmland_tags,
        )
        osm_farmland = raw_farm[raw_farm.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        osm_farmland.to_file(CACHE_FARM_PATH, driver="GeoJSON")
        print(f"  Cached {len(osm_farmland)} farmland polygons to {CACHE_FARM_PATH}")

    print(f"  Farmland polygons:   {len(osm_farmland)}")

    # Step 3: Compute continuous distance columns
    print("\n" + "=" * 72)
    print("STEP 3: Computing distance_to_industrial_m and distance_to_farmland_m")
    print("=" * 72)

    if len(osm_industrial) > 0:
        ind_utm = osm_industrial.to_crs(UTM_CRS)
        ind_union = unary_union(ind_utm.geometry.values)
        gdf["distance_to_industrial_m"] = gdf_utm.geometry.distance(ind_union).round(1)
    else:
        gdf["distance_to_industrial_m"] = np.nan

    if len(osm_farmland) > 0:
        farm_utm = osm_farmland.to_crs(UTM_CRS)
        farm_union = unary_union(farm_utm.geometry.values)
        gdf["distance_to_farmland_m"] = gdf_utm.geometry.distance(farm_union).round(1)
    else:
        gdf["distance_to_farmland_m"] = np.nan

    print(f"  distance_to_industrial_m -- min: {gdf['distance_to_industrial_m'].min():.1f}m, "
          f"median: {gdf['distance_to_industrial_m'].median():.1f}m, max: {gdf['distance_to_industrial_m'].max():.1f}m")
    print(f"  distance_to_farmland_m   -- min: {gdf['distance_to_farmland_m'].min():.1f}m, "
          f"median: {gdf['distance_to_farmland_m'].median():.1f}m, max: {gdf['distance_to_farmland_m'].max():.1f}m")

    # Step 4: Per-Cluster Historical Statistics
    print("\n" + "=" * 72)
    print("STEP 4: Computing per-cluster historical statistics")
    print("=" * 72)

    clustered_mask = gdf["cluster_id"] != -1
    clustered = gdf.loc[clustered_mask]

    cluster_stats = clustered.groupby("cluster_id")["frp"].agg(
        historical_mean_frp="mean",
        historical_max_frp="max",
        historical_std_frp="std",
    )

    gdf["historical_mean_frp"] = gdf["cluster_id"].map(cluster_stats["historical_mean_frp"])
    gdf["historical_max_frp"]  = gdf["cluster_id"].map(cluster_stats["historical_max_frp"])
    gdf["historical_std_frp"]  = gdf["cluster_id"].map(cluster_stats["historical_std_frp"])

    gdf["frp_deviation"] = np.where(
        clustered_mask,
        (gdf["frp"] / gdf["historical_mean_frp"]).round(4),
        np.nan
    )

    cluster_night_ratio = (
        clustered
        .groupby("cluster_id")["daynight"]
        .apply(lambda s: (s == "N").mean())
        .rename("night_detection_ratio")
    )
    gdf["night_detection_ratio"] = gdf["cluster_id"].map(cluster_night_ratio)

    noise_mask = gdf["cluster_id"] == -1
    gdf.loc[noise_mask, [
        "historical_mean_frp", "historical_max_frp", "historical_std_frp",
        "frp_deviation", "night_detection_ratio"
    ]] = np.nan

    # Step 5: Expanded 5-Class Classification -> class_label
    print("\n" + "=" * 72)
    print("STEP 5: Running 5-class classification logic")
    print("=" * 72)

    gdf["class_label"] = "Unknown"

    # Rule 1: Industrial
    near_industrial = gdf["distance_to_industrial_m"] <= 1000
    type_is_2 = gdf["type"] == 2
    high_persistence = gdf["distinct_days"] >= 10
    industrial_rule = near_industrial & (type_is_2 | high_persistence)
    gdf.loc[industrial_rule, "class_label"] = "Industrial"

    # Rule 2: Agricultural Burning (distance_to_farmland_m <= 2000m)
    near_farmland = gdf["distance_to_farmland_m"] <= 2000
    not_near_industrial = gdf["distance_to_industrial_m"] > 1000
    low_persistence = gdf["distinct_days"] <= 3
    ag_burn_rule = near_farmland & not_near_industrial & low_persistence

    still_unknown = gdf["class_label"] == "Unknown"
    gdf.loc[still_unknown & ag_burn_rule, "class_label"] = "Agricultural Burning"

    # Rule 3: Natural Fire
    type_is_0 = gdf["type"] == 0
    still_unknown = gdf["class_label"] == "Unknown"
    gdf.loc[still_unknown & type_is_0 & not_near_industrial, "class_label"] = "Natural Fire"

    # Rule 4: Unknown remains

    print("  Punjab class_label breakdown:")
    for label, count in gdf["class_label"].value_counts().sort_index().items():
        pct = count / len(gdf) * 100
        print(f"    {label:25s} : {count:5,}  ({pct:5.1f}%)")

    # Step 6: Sentinel-2 placeholders
    for col in ["ndvi", "ndbi", "ndwi"]:
        gdf[col] = "unavailable"

    # Step 7: Alert Flags
    # Alert 1: Anomaly High FRP (> 3x cluster mean)
    gdf["alert_high_frp"] = (gdf["frp_deviation"] >= 3.0).astype(int)
    # Alert 2: Anomaly New Clustered Source
    gdf["alert_new_cluster"] = 0

    # Step 8: Save to feature_vector_punjab.csv
    print("\n" + "=" * 72)
    print("STEP 8: Saving data_punjab/feature_vector_punjab.csv")
    print("=" * 72)
    out_df = gdf.drop(columns=["geometry"])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Successfully wrote {len(out_df):,} rows x {len(out_df.columns)} columns to:")
    print(f"  --> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
