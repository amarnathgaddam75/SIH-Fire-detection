"""
classify_hotspots_punjab.py
===========================
FIRMS Fire-vs-Industrial Hotspot Classifier for Punjab Stubble-Burning Belt.

Bounding box: North 30.6, South 29.6, East 76.5, West 75.0
Projected CRS: EPSG:32643 (UTM Zone 43N)

Inputs:  data_punjab/DL_FIRE_J1V-C2_800876/fire_archive_J1V-C2_800876.csv
Outputs: data_punjab/firms_labeled_punjab.csv
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_punjab")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Punjab Bounding Box
BBOX_NORTH = 30.6
BBOX_SOUTH = 29.6
BBOX_EAST  = 76.5
BBOX_WEST  = 75.0

# UTM zone 43N (appropriate for 72E - 78E)
UTM_CRS = "EPSG:32643"

OUTPUT_LABELED = os.path.join(DATA_DIR, "firms_labeled_punjab.csv")
OSM_CACHE_PATH = os.path.join(CACHE_DIR, "osm_industrial_polygons_punjab.geojson")


def main():
    print("=" * 72)
    print("STEP 1: Loading FIRMS CSV data for Punjab")
    print("=" * 72)

    # Search for archive and nrt CSVs in data_punjab
    archive_csvs = glob.glob(os.path.join(DATA_DIR, "**", "*archive*.csv"), recursive=True)
    nrt_csvs = glob.glob(os.path.join(DATA_DIR, "**", "*nrt*.csv"), recursive=True)

    dfs = []
    for f in archive_csvs:
        df_a = pd.read_csv(f)
        print(f"  Loaded archive: {f} ({len(df_a):,} rows)")
        dfs.append(df_a)

    for f in nrt_csvs:
        df_n = pd.read_csv(f)
        df_n["type"] = np.nan
        print(f"  Loaded NRT: {f} ({len(df_n):,} rows)")
        dfs.append(df_n)

    if not dfs:
        raise FileNotFoundError(f"No FIRMS CSV files found in {DATA_DIR}")

    df = pd.concat(dfs, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df.sort_values("acq_date").reset_index(drop=True)

    print(f"  Combined Punjab data: {len(df):,} rows, "
          f"{df['acq_date'].min().date()} -> {df['acq_date'].max().date()}\n")

    # Step 2: Build GeoDataFrame
    print("STEP 2: Building GeoDataFrame")
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    print(f"  GeoDataFrame with {len(gdf)} points, CRS = {gdf.crs}\n")

    # Step 3: Query / Load OSM Industrial Polygons
    print("STEP 3: Querying/Loading OSM for industrial features in Punjab bbox...")
    osm_tags = {
        "landuse": ["industrial", "quarry"],
        "power": "plant",
        "man_made": ["works", "mineshaft"],
    }

    if os.path.exists(OSM_CACHE_PATH):
        print(f"  Loading cached industrial polygons from {OSM_CACHE_PATH}")
        osm_polygons = gpd.read_file(OSM_CACHE_PATH)
    else:
        try:
            print(f"  Querying Overpass for bbox: ({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH})")
            osm_gdf = ox.features_from_bbox(
                bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
                tags=osm_tags,
            )
            osm_polygons = osm_gdf[
                osm_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
            ].copy()
            osm_polygons.to_file(OSM_CACHE_PATH, driver="GeoJSON")
            print(f"  Saved {len(osm_polygons)} industrial polygons to cache: {OSM_CACHE_PATH}")
        except Exception as e:
            print(f"  Warning: OSM query failed: {e}")
            osm_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    print(f"  Usable industrial polygons: {len(osm_polygons)}\n")

    # Step 4: Spatial Join
    print("STEP 4: Spatial join — classifying hotspots by industrial proximity")
    gdf["label"] = "unclassified"

    if len(osm_polygons) > 0:
        # Direct containment
        joined_direct = gpd.sjoin(gdf, osm_polygons, how="inner", predicate="within")
        direct_indices = joined_direct.index.unique()
        gdf.loc[direct_indices, "label"] = "industrial_direct"
        print(f"  industrial_direct : {len(direct_indices):,} hotspots inside polygon")

        # 1 km buffer in UTM
        osm_utm = osm_polygons.to_crs(UTM_CRS)
        osm_buffered = osm_utm.copy()
        osm_buffered["geometry"] = osm_utm.geometry.buffer(1000)
        osm_buffered = osm_buffered.to_crs("EPSG:4326")

        joined_nearby = gpd.sjoin(gdf, osm_buffered, how="inner", predicate="within")
        nearby_indices = joined_nearby.index.unique()
        nearby_only = nearby_indices.difference(direct_indices)
        gdf.loc[nearby_only, "label"] = "industrial_nearby"
        print(f"  industrial_nearby : {len(nearby_only):,} hotspots within 1km buffer")

    unclass_count = (gdf["label"] == "unclassified").sum()
    print(f"  unclassified      : {unclass_count:,} hotspots\n")

    # Step 5: Persistence clustering (DBSCAN, eps=500m)
    print("STEP 5: Persistence clustering (DBSCAN, eps=500m, UTM Zone 43N)")
    gdf_utm = gdf.to_crs(UTM_CRS)
    coords = np.column_stack([gdf_utm.geometry.x, gdf_utm.geometry.y])

    clustering = DBSCAN(eps=500, min_samples=2).fit(coords)
    gdf["cluster_id"] = clustering.labels_

    n_clusters = len(set(clustering.labels_) - {-1})
    n_noise = (clustering.labels_ == -1).sum()
    print(f"  Found {n_clusters} clusters and {n_noise:,} noise points")

    cluster_day_counts = (
        gdf[gdf["cluster_id"] != -1]
        .groupby("cluster_id")["acq_date"]
        .nunique()
        .rename("distinct_days")
    )
    gdf["distinct_days"] = gdf["cluster_id"].map(cluster_day_counts).fillna(0).astype(int)

    # Step 6: Summary & Export
    print("\n" + "=" * 72)
    print("STEP 6: Saving firms_labeled_punjab.csv")
    print("=" * 72)
    out_df = gdf.drop(columns=["geometry"])
    out_df.to_csv(OUTPUT_LABELED, index=False)
    print(f"  Saved {len(out_df):,} rows to {OUTPUT_LABELED}")
    print("\nLabel distribution:")
    for l, c in out_df["label"].value_counts().items():
        print(f"    {l:25s} : {c:,}")


if __name__ == "__main__":
    main()
