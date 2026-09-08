"""
fetch_sentinel_pilot.py
=======================
Queries the Sentinel Hub Statistical API on Copernicus Data Space Ecosystem (CDSE)
for a pilot subset of thermal hotspots from data/feature_vector.csv:
  - Top 5 rows by distinct_days (persistent industrial/flare sites)
  - Top 10 rows with alert_flag == "Abnormal spike" (by frp_deviation)
  - 5 rows with class_label == "Natural Fire" (for environmental contrast)

For each point:
  - Bounding box: 500m x 500m in EPSG:32644 (UTM 44N)
  - Data collection: Sentinel-2 L2A, max cloud cover 30%
  - Aggregation window: 30 days surrounding each row's acq_date (±15 days)
  - Computes mean NDVI, NDBI, and NDWI via Statistical API evalscript
  - Updates only the pilot rows in data/feature_vector.csv
"""

import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pyproj import Transformer
from sentinelhub import SHConfig, SentinelHubSession

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
PILOT_EXPORT_PATH = os.path.join(DATA_DIR, "sentinel2_pilot_results.csv")
STAT_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

# Evalscript to compute NDVI, NDBI, and NDWI from Sentinel-2 L2A
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B08", "B11", "dataMask", "SCL"]
    }],
    output: [
      { id: "default", bands: 3, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  let denom_ndvi = sample.B08 + sample.B04;
  let ndvi = denom_ndvi > 0 ? (sample.B08 - sample.B04) / denom_ndvi : 0;

  let denom_ndbi = sample.B11 + sample.B08;
  let ndbi = denom_ndbi > 0 ? (sample.B11 - sample.B08) / denom_ndbi : 0;

  let denom_ndwi = sample.B03 + sample.B08;
  let ndwi = denom_ndwi > 0 ? (sample.B03 - sample.B08) / denom_ndwi : 0;

  // SCL cloud masking: 3=cloud shadows, 8=med prob, 9=high prob, 10=thin cirrus
  let isCloud = [3, 8, 9, 10].includes(sample.SCL);
  let valid = (sample.dataMask === 1 && !isCloud) ? 1 : 0;

  return {
    default: [ndvi, ndbi, ndwi],
    dataMask: [valid]
  };
}
"""


def select_pilot_indices(df):
    """Select the requested pilot subset of rows."""
    # 1. Top 5 rows by distinct_days
    top_distinct = df.sort_values(by=["distinct_days", "frp"], ascending=[False, False]).head(5)

    # 2. alert_flag == "Abnormal spike" (top 10 by frp_deviation if > 10)
    spikes = df[df["alert_flag"] == "Abnormal spike"]
    if len(spikes) > 10:
        spikes = spikes.sort_values(by="frp_deviation", ascending=False).head(10)

    # 3. 5 rows with class_label == "Natural Fire"
    natural = df[df["class_label"] == "Natural Fire"].head(5)

    # Combine indices while preserving category metadata
    pilot_dict = {}
    for idx in top_distinct.index:
        pilot_dict[idx] = "Top Distinct Days"
    for idx in spikes.index:
        pilot_dict[idx] = "Abnormal FRP Spike"
    for idx in natural.index:
        pilot_dict[idx] = "Natural Fire Baseline"

    return pilot_dict


def query_statistical_api(lat, lon, acq_date_str, token, transformer):
    """Query Statistical API for a 500m x 500m box over 30 days surrounding acq_date."""
    # Compute dates: ±15 days around acq_date
    acq_date = datetime.strptime(str(acq_date_str)[:10], "%Y-%m-%d")
    start_date = (acq_date - timedelta(days=15)).strftime("%Y-%m-%dT00:00:00Z")
    end_date = (acq_date + timedelta(days=15)).strftime("%Y-%m-%dT23:59:59Z")

    # Compute 500m x 500m bounding box in UTM 44N
    x, y = transformer.transform(lon, lat)
    utm_bbox = [round(x - 250, 2), round(y - 250, 2), round(x + 250, 2), round(y + 250, 2)]

    payload = {
        "input": {
            "bounds": {
                "bbox": utm_bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32644"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": start_date,
                        "to": end_date
                    },
                    "maxCloudCoverage": 30
                }
            }]
        },
        "aggregation": {
            "timeRange": {
                "from": start_date,
                "to": end_date
            },
            "aggregationInterval": {
                "of": "P30D"
            },
            "evalscript": EVALSCRIPT,
            "resx": 10,
            "resy": 10
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(STAT_URL, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"    [API Error {response.status_code}]: {response.text}")
        return None, None, None

    res_data = response.json()
    try:
        data_intervals = res_data.get("data", [])
        if not data_intervals:
            return None, None, None

        bands = data_intervals[0]["outputs"]["default"]["bands"]
        ndvi_mean = bands["B0"]["stats"]["mean"]
        ndbi_mean = bands["B1"]["stats"]["mean"]
        ndwi_mean = bands["B2"]["stats"]["mean"]

        return round(float(ndvi_mean), 4), round(float(ndbi_mean), 4), round(float(ndwi_mean), 4)
    except Exception as e:
        print(f"    [Parse Error]: {e}")
        return None, None, None


def main():
    print("=" * 80)
    print("GeoFlare -- Sentinel-2 Statistical API Pilot Ingestion (CDSE)")
    print("=" * 80)

    # 1. Authenticate with Sentinel Hub CDSE profile
    print("\n1. Authenticating with Copernicus Data Space Ecosystem (profile: 'cdse')...")
    config = SHConfig("cdse")
    session = SentinelHubSession(config=config)
    token = session.token.get("access_token")
    if not token:
        raise ValueError("Failed to retrieve valid access token from CDSE.")
    print("   Authentication successful. Active OAuth bearer token established.")

    # 2. Load dataset and select pilot subset
    print(f"\n2. Loading {CSV_PATH} and selecting pilot subset...")
    df = pd.read_csv(CSV_PATH)
    pilot_dict = select_pilot_indices(df)
    pilot_indices = sorted(list(pilot_dict.keys()))
    print(f"   Selected {len(pilot_indices)} unique pilot points:")
    for idx in pilot_indices:
        r = df.iloc[idx]
        reason = pilot_dict[idx]
        print(f"     - Row {idx:4d}: ({r['latitude']:.4f}, {r['longitude']:.4f}) | {r['acq_date']} | "
              f"FRP: {r['frp']:5.1f} MW | Class: {r['class_label']:12s} | Reason: {reason}")

    # 3. Setup Coordinate Transformer (WGS84 -> UTM 44N)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)

    # Ensure ndvi, ndbi, ndwi columns exist and allow object/float values
    for col in ["ndvi", "ndbi", "ndwi"]:
        if col not in df.columns:
            df[col] = "unavailable"
        else:
            df[col] = df[col].astype(object)

    # 4. Process each pilot point
    print(f"\n3. Querying Sentinel Hub Statistical API for {len(pilot_indices)} points (500m x 500m, 30-day window)...")
    results = []

    for i, idx in enumerate(pilot_indices, 1):
        row = df.iloc[idx]
        lat = row["latitude"]
        lon = row["longitude"]
        date_str = str(row["acq_date"])
        category = pilot_dict[idx]
        class_lbl = row["class_label"]
        frp = row["frp"]

        print(f"\n   [{i:2d}/{len(pilot_indices)}] Row {idx}: Lat {lat:.4f}, Lon {lon:.4f}, Date {date_str} ({category})...")

        ndvi, ndbi, ndwi = query_statistical_api(lat, lon, date_str, token, transformer)

        if ndvi is not None:
            print(f"        -> SUCCESS: NDVI={ndvi:+.4f} | NDBI={ndbi:+.4f} | NDWI={ndwi:+.4f}")
            df.at[idx, "ndvi"] = ndvi
            df.at[idx, "ndbi"] = ndbi
            df.at[idx, "ndwi"] = ndwi
            status = "Success"
        else:
            print(f"        -> NO DATA or Cloud Limit Exceeded")
            status = "No Data"

        results.append({
            "row_index": idx,
            "latitude": lat,
            "longitude": lon,
            "acq_date": date_str,
            "class_label": class_lbl,
            "frp": frp,
            "pilot_category": category,
            "ndvi": ndvi,
            "ndbi": ndbi,
            "ndwi": ndwi,
            "status": status
        })

        # Polite rate limiting between queries
        time.sleep(0.3)

    # 5. Save updated feature_vector.csv
    print(f"\n4. Saving updated dataset to {CSV_PATH}...")
    df.to_csv(CSV_PATH, index=False)
    print(f"   Successfully updated feature_vector.csv ({len(df):,} rows). Only pilot rows modified.")

    # 6. Save pilot results table for review
    results_df = pd.DataFrame(results)
    results_df.to_csv(PILOT_EXPORT_PATH, index=False)
    print(f"   Pilot summary exported to: {PILOT_EXPORT_PATH}")

    # 7. Print summary table
    print("\n" + "=" * 80)
    print("PILOT SATELLITE INDICES SUMMARY TABLE:")
    print("=" * 80)
    display_cols = ["row_index", "acq_date", "class_label", "pilot_category", "ndvi", "ndbi", "ndwi"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
