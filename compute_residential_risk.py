"""
compute_residential_risk.py
===========================
1. Pulls/loads OpenStreetMap residential polygons (landuse=residential) for the
   Korba bounding box in UTM 44N (EPSG:32644).
2. Computes distance_to_residential_m for every row in data/feature_vector.csv.
3. Implements an explainable point-based risk scoring system:
     - Factor 1: class_label / industrial proximity (0 to 3 points)
     - Factor 2: alert_flag / abnormal FRP spike (0 to 3 points)
     - Factor 3: distance_to_residential_m proximity (0 to 3 points)
   Outputs four risk tiers:
     - "Critical": Score >= 7 (Abnormal spikes near residential settlements)
     - "High":     Score 5 - 6 (Active industrial / spikes in close proximity)
     - "Medium":   Score 3 - 4 (Moderate distance or low-intensity fires near homes)
     - "Low":      Score 0 - 2 (Distant or baseline natural fires)
4. Prints the distribution and all rows that land in "Critical".
5. Saves the updated dataset back to data/feature_vector.csv and exports risk_score
   to data/firms_final_ml.geojson and data/firms_final.geojson.
"""

import os
import json
import warnings
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
GEOJSON_ML_PATH = os.path.join(DATA_DIR, "firms_final_ml.geojson")
GEOJSON_FINAL_PATH = os.path.join(DATA_DIR, "firms_final.geojson")
CACHE_RES_PATH = os.path.join(CACHE_DIR, "osm_residential_polygons.geojson")

BBOX_NORTH = 22.6
BBOX_SOUTH = 22.1
BBOX_EAST  = 83.1
BBOX_WEST  = 82.3
UTM_CRS    = "EPSG:32644"


def get_residential_polygons():
    """Retrieve or load cached OSM residential polygons."""
    if os.path.exists(CACHE_RES_PATH):
        print(f"  Loading cached residential polygons from {CACHE_RES_PATH}...")
        res_gdf = gpd.read_file(CACHE_RES_PATH)
    else:
        print(f"  Querying Overpass for landuse=residential in Korba bbox ({BBOX_WEST}, {BBOX_SOUTH}, {BBOX_EAST}, {BBOX_NORTH})...")
        res_tags = {"landuse": "residential"}
        raw_res = ox.features_from_bbox(
            bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
            tags=res_tags
        )
        res_gdf = raw_res[raw_res.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        res_gdf.to_file(CACHE_RES_PATH, driver="GeoJSON")
        print(f"  Saved {len(res_gdf)} residential polygons to {CACHE_RES_PATH}")

    return res_gdf


def calculate_risk_points(row):
    """
    Explainable point-based risk scoring function combining 3 factors:
    1. Class Hazard (0 - 3 pts):
       - Industrial / Unknown inside industrial buffer (<=1000m): 3 pts
       - Unknown outside industrial buffer: 1 pt
       - Agricultural Burning: 1 pt
       - Natural Fire: 0 pts
    2. Operational Spike Hazard (0 - 3 pts):
       - Abnormal spike (alert_flag == 'Abnormal spike' or frp_deviation >= 2.5): 3 pts
       - Normal: 0 pts
    3. Population Proximity Hazard (0 - 3 pts):
       - <= 500m to homes: 3 pts
       - 500m - 1,500m to homes: 2 pts
       - 1,500m - 3,000m to homes: 1 pt
       - > 3,000m to homes: 0 pts
    """
    # Factor 1: Class Hazard
    lbl = str(row.get("class_label", ""))
    dist_ind = row.get("distance_to_industrial_m", 99999)
    if lbl == "Industrial" or (lbl == "Unknown" and dist_ind <= 1000):
        pts_class = 3
    elif lbl in ["Unknown", "Agricultural Burning"]:
        pts_class = 1
    else:
        pts_class = 0

    # Factor 2: Spike Hazard
    alert = str(row.get("alert_flag", ""))
    dev = row.get("frp_deviation", 0.0)
    if alert == "Abnormal spike" or (pd.notna(dev) and float(dev) >= 2.5):
        pts_alert = 3
    else:
        pts_alert = 0

    # Factor 3: Proximity to Residential Settlements
    d_res = row.get("distance_to_residential_m", 99999)
    if d_res <= 500:
        pts_dist = 3
    elif d_res <= 1500:
        pts_dist = 2
    elif d_res <= 3000:
        pts_dist = 1
    else:
        pts_dist = 0

    total = pts_class + pts_alert + pts_dist

    # Map to explainable categories
    if total >= 7:
        tier = "Critical"
    elif total >= 5:
        tier = "High"
    elif total >= 3:
        tier = "Medium"
    else:
        tier = "Low"

    return total, tier


def main():
    print("=" * 80)
    print("GeoFlare -- Residential Distance & Multi-Factor Risk Assessment")
    print("=" * 80)

    # 1. Load feature_vector.csv
    print(f"\n1. Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Loaded {len(df):,} rows.")

    # 2. Get residential polygons and compute distances in UTM 44N
    print("\n2. Processing OSM Residential Polygons in UTM Zone 44N (EPSG:32644)...")
    res_gdf = get_residential_polygons().to_crs(UTM_CRS)
    res_union = unary_union(res_gdf.geometry.values)

    # Reproject hotspot points
    geom = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf_utm = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326").to_crs(UTM_CRS)

    print("   Calculating distance_to_residential_m for all 2,008 points...")
    df["distance_to_residential_m"] = gdf_utm.geometry.distance(res_union).round(1)

    min_d = df["distance_to_residential_m"].min()
    med_d = df["distance_to_residential_m"].median()
    max_d = df["distance_to_residential_m"].max()
    print(f"   distance_to_residential_m -- Min: {min_d:.1f}m | Median: {med_d:.1f}m | Max: {max_d:.1f}m")

    # 3. Compute risk scores
    print("\n3. Calculating point-based risk scores...")
    risk_results = [calculate_risk_points(row) for _, row in df.iterrows()]
    df["risk_points"] = [r[0] for r in risk_results]
    df["risk_score"] = [r[1] for r in risk_results]

    # Print distribution
    print("\n" + "-" * 45)
    print("  RISK SCORE DISTRIBUTION")
    print("-" * 45)
    score_order = ["Critical", "High", "Medium", "Low"]
    for tier in score_order:
        cnt = (df["risk_score"] == tier).sum()
        pct = (cnt / len(df)) * 100
        print(f"  {tier:12s} : {cnt:5,d}  ({pct:5.1f}%)")
    print("-" * 45)
    print(f"  {'TOTAL':12s} : {len(df):5,d}  (100.0%)\n")

    # 4. Print Critical rows for sanity check
    critical_df = df[df["risk_score"] == "Critical"].sort_values(
        by=["risk_points", "frp"], ascending=[False, False]
    )
    print("=" * 80)
    print(f"CRITICAL RISK HOTSPOTS SANITY CHECK ({len(critical_df)} rows)")
    print("=" * 80)
    display_cols = [
        "latitude", "longitude", "acq_date", "frp", "frp_deviation",
        "class_label", "alert_flag", "distance_to_residential_m", "risk_points"
    ]
    print(critical_df[display_cols].to_string(index=False))
    print("=" * 80)

    # 5. Save back to feature_vector.csv
    print(f"\n5. Updating {CSV_PATH}...")
    df.to_csv(CSV_PATH, index=False)
    print(f"   Saved {len(df):,} rows with new columns: distance_to_residential_m, risk_score.")

    # 6. Update GeoJSON files with risk_score and distance_to_residential_m
    for geo_path in [GEOJSON_ML_PATH, GEOJSON_FINAL_PATH]:
        if os.path.exists(geo_path):
            print(f"\n6. Adding risk_score and distance_to_residential_m to {os.path.basename(geo_path)}...")
            with open(geo_path, "r", encoding="utf-8") as f:
                geo_data = json.load(f)

            for i, feat in enumerate(geo_data.get("features", [])):
                if i < len(df):
                    row = df.iloc[i]
                    feat["properties"]["distance_to_residential_m"] = float(row["distance_to_residential_m"])
                    feat["properties"]["risk_score"] = str(row["risk_score"])
                    feat["properties"]["risk_points"] = int(row["risk_points"])

            with open(geo_path, "w", encoding="utf-8") as f:
                json.dump(geo_data, f, indent=2)
            print(f"   Updated {len(geo_data.get('features', [])):,} features in {os.path.basename(geo_path)}.")

    print("\n" + "=" * 80)
    print("Residential risk evaluation and export complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
