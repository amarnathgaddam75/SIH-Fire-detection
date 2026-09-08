"""
api.py
======
FastAPI Backend for GeoFlare Satellite Hotspot Classification System.

Endpoints:
1. GET /hotspots            - Serves data/firms_final_ml.geojson directly.
2. GET /alerts              - Returns rows where risk_score == "Critical", sorted
                              by acq_date descending, with precomputed narrative text fields.
3. POST /predict/existing   - Given a row index or coordinates matching feature_vector.csv,
                              evaluates live XGBoost prediction and computes top 3 SHAP drivers.
4. POST /predict/hypothetical - Given arbitrary lat, lon, and frp, computes distance to cached
                              OSM industrial/farmland polygons, applies median imputation for
                              missing values, and returns live prediction with top 3 SHAP drivers.
"""

import os
import json
import warnings
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import shap
from shapely.geometry import Point
from shapely.ops import unary_union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Setup Directories and Artifact Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

GEOJSON_PATH = os.path.join(DATA_DIR, "firms_final_ml.geojson")
GEOJSON_FALLBACK = os.path.join(DATA_DIR, "firms_final.geojson")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
MODEL_PATH = os.path.join(DATA_DIR, "classifier.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "model_features.txt")

IND_CACHE_PATH = os.path.join(CACHE_DIR, "osm_industrial_polygons.geojson")
FARM_CACHE_PATH = os.path.join(CACHE_DIR, "osm_farmland_polygons.geojson")

FEATURE_COLUMNS = [
    "frp",
    "brightness",
    "confidence",
    "daynight",
    "distance_to_industrial_m",
    "distance_to_farmland_m",
    "distinct_days",
    "historical_mean_frp",
    "historical_max_frp",
    "historical_std_frp",
    "frp_deviation",
    "night_detection_ratio",
    "ndvi",
    "ndbi",
    "ndwi",
]

# ──────────────────────────────────────────────────────────────────────────────
# Initialize FastAPI App & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoFlare Hotspot Intelligence API",
    description="FastAPI service for NASA FIRMS satellite fire classification, early-warning escalation, and explainable ML inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Global In-Memory Model, SHAP Explainer, and Cached Polygons
# ──────────────────────────────────────────────────────────────────────────────
print("Loading model, features, and cached spatial data...")

# 1. Load ML Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Classifier model missing at {MODEL_PATH}")
classifier_model = joblib.load(MODEL_PATH)

# 2. Load Model Feature Names (23 features)
with open(FEATURES_PATH, "r") as f:
    EXPECTED_FEATURES = [line.strip() for line in f if line.strip()]

# 3. Load Feature Table & Training Medians for Imputation
df_feature_vector = pd.read_csv(CSV_PATH)

# Determine class names from model or dataset
CLASSES = ["Agricultural Burning", "Industrial", "Natural Fire", "Unknown"]
if hasattr(classifier_model, "classes_"):
    # If model stores integer classes or strings
    if isinstance(classifier_model.classes_[0], (int, np.integer)):
        pass # Will map through sorted unique class_label in df
    else:
        CLASSES = list(classifier_model.classes_)

# Precompute training medians for imputation
TRAINING_MEDIANS = {}
for col in FEATURE_COLUMNS:
    series = pd.to_numeric(df_feature_vector[col], errors="coerce")
    med = series.median()
    TRAINING_MEDIANS[col] = 0.0 if pd.isna(med) else float(med)

# 4. Initialize SHAP TreeExplainer
print("Initializing SHAP TreeExplainer...")
tree_explainer = shap.TreeExplainer(classifier_model)

# 5. Load / Cache OSM Polygons for Instant Geometry Distance Math
print("Loading cached OSM geometry unions in UTM Zone 44N (EPSG:32644)...")
if not os.path.exists(IND_CACHE_PATH) or not os.path.exists(FARM_CACHE_PATH):
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = CACHE_DIR
    BBOX_NORTH, BBOX_SOUTH, BBOX_EAST, BBOX_WEST = 22.6, 22.1, 83.1, 82.3

    if not os.path.exists(IND_CACHE_PATH):
        raw_ind = ox.features_from_bbox(
            bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
            tags={"landuse": ["industrial", "quarry"], "power": "plant", "man_made": ["works", "mineshaft"]},
        )
        ind_polys = raw_ind[raw_ind.geometry.type.isin(["Polygon", "MultiPolygon"])][["geometry"]].copy()
        ind_polys.to_file(IND_CACHE_PATH, driver="GeoJSON")

    if not os.path.exists(FARM_CACHE_PATH):
        raw_farm = ox.features_from_bbox(
            bbox=(BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH),
            tags={"landuse": ["farmland", "agricultural"]},
        )
        farm_polys = raw_farm[raw_farm.geometry.type.isin(["Polygon", "MultiPolygon"])][["geometry"]].copy()
        farm_polys.to_file(FARM_CACHE_PATH, driver="GeoJSON")

gdf_industrial = gpd.read_file(IND_CACHE_PATH).to_crs(epsg=32644)
industrial_union = unary_union(gdf_industrial.geometry)

gdf_farmland = gpd.read_file(FARM_CACHE_PATH).to_crs(epsg=32644)
farmland_union = unary_union(gdf_farmland.geometry)
print("GeoFlare API engine initialized successfully!")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def format_feature_value(feat_name: str, val: float) -> str:
    """Formats numeric values into human-readable strings with intuitive units."""
    if "distance" in feat_name:
        return f"{val:,.0f} m"
    elif feat_name in ["frp", "historical_mean_frp", "historical_max_frp", "historical_std_frp"]:
        return f"{val:.1f} MW"
    elif feat_name == "brightness":
        return f"{val:.1f} K"
    elif feat_name == "distinct_days":
        return f"{int(val)} days"
    elif feat_name == "night_detection_ratio":
        return f"{val * 100:.1f}% night"
    elif feat_name == "daynight":
        return "Day" if val == 1.0 else "Night"
    elif feat_name == "confidence":
        mapping = {0.0: "Low (0)", 1.0: "Nominal (1)", 2.0: "High (2)"}
        return mapping.get(val, f"{val:.0f}")
    elif "_was_missing" in feat_name:
        return "Missing (Imputed)" if val == 1.0 else "Observed"
    elif feat_name == "frp_deviation":
        return f"{val:.2f}x baseline"
    else:
        return f"{val:.2f}"


def build_model_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw columns into the exact 23-feature model input matrix
    with imputation and companion _was_missing flags.
    """
    df = df_raw.copy()

    # Encode daynight
    if "daynight" in df.columns:
        df["daynight"] = df["daynight"].map({"D": 1.0, "N": 0.0}).fillna(
            pd.to_numeric(df["daynight"], errors="coerce")
        )
    else:
        df["daynight"] = 1.0

    # Encode confidence
    if "confidence" in df.columns:
        conf_map = {"l": 0.0, "n": 1.0, "h": 2.0}
        df["confidence"] = df["confidence"].map(conf_map).fillna(
            pd.to_numeric(df["confidence"], errors="coerce")
        )
    else:
        df["confidence"] = 1.0

    # Numeric indices
    for col in ["ndvi", "ndbi", "ndwi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    X = pd.DataFrame(index=df.index)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
        else:
            series = pd.Series(np.nan, index=df.index)

        # Companion was_missing flag
        flag_col = f"{col}_was_missing"
        if flag_col in EXPECTED_FEATURES:
            X[flag_col] = series.isnull().astype(int)

        # Fill with precomputed training median
        X[col] = series.fillna(TRAINING_MEDIANS.get(col, 0.0))

    # Ensure columns strictly match order
    for col in EXPECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    return X[EXPECTED_FEATURES]


def compute_prediction_and_shap(X_matrix: pd.DataFrame, target_index: int = 0) -> Dict[str, Any]:
    """Runs classifier.pkl inference and computes top 3 SHAP contributing features."""
    probs = classifier_model.predict_proba(X_matrix)[target_index]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {c: float(probs[i]) for i, c in enumerate(CLASSES)}

    # Compute SHAP values for this single sample
    shap_vals = tree_explainer.shap_values(X_matrix.iloc[[target_index]])
    shap_arr = np.array(shap_vals)  # Shape: (1, 23, 4)

    # SHAP values for the predicted class
    class_shap = shap_arr[0, :, pred_idx]

    # Rank features by absolute impact
    top_indices = np.argsort(np.abs(class_shap))[::-1][:3]

    top_features = []
    for rank, feat_idx in enumerate(top_indices, 1):
        feat_name = EXPECTED_FEATURES[feat_idx]
        raw_val = float(X_matrix.iloc[target_index, feat_idx])
        shap_val = float(class_shap[feat_idx])
        formatted_val = format_feature_value(feat_name, raw_val)

        direction = "TOWARD" if shap_val > 0 else "AWAY"
        clean_name = feat_name.replace("_", " ").title()
        if "_was_missing" in feat_name:
            clean_name = f"{feat_name.replace('_was_missing', '').replace('_', ' ').title()} Missing Flag"

        narrative = (
            f"{clean_name} = {formatted_val} pushed prediction {direction} {pred_class} ({shap_val:+.2f})"
        )

        top_features.append({
            "rank": rank,
            "feature": feat_name,
            "display_name": clean_name,
            "value": formatted_val,
            "raw_value": raw_val,
            "shap_value": round(shap_val, 3),
            "direction": direction,
            "narrative": narrative
        })

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "class_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "top_shap_features": top_features
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Request Models
# ──────────────────────────────────────────────────────────────────────────────
class ExistingPredictionRequest(BaseModel):
    row_index: Optional[int] = Field(None, description="0-indexed row number from feature_vector.csv")
    latitude: Optional[float] = Field(None, description="Hotspot latitude")
    longitude: Optional[float] = Field(None, description="Hotspot longitude")


class HypotheticalPredictionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude of hypothetical fire/flare")
    longitude: float = Field(..., description="Longitude of hypothetical fire/flare")
    frp: float = Field(..., ge=0.0, description="Fire Radiative Power in MW")
    daynight: Optional[str] = Field("D", description="'D' for daytime, 'N' for nighttime pass")
    confidence: Optional[str] = Field("n", description="Satellite confidence: 'l', 'n', or 'h'")


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Service health verification."""
    return {"status": "ok", "service": "GeoFlare Hotspot Intelligence API", "features_loaded": len(EXPECTED_FEATURES)}


@app.get("/hotspots")
def get_hotspots():
    """
    1. GET /hotspots
    Returns the contents of data/firms_final_ml.geojson directly (or base geojson fallback).
    """
    target_path = GEOJSON_PATH if os.path.exists(GEOJSON_PATH) else GEOJSON_FALLBACK
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Hotspots GeoJSON file not found.")

    return FileResponse(target_path, media_type="application/geo+json")


@app.get("/alerts")
def get_alerts():
    """
    2. GET /alerts
    Returns rows where risk_score == 'Critical', sorted by acq_date descending,
    with narrative text fields ready to display.
    """
    critical_df = df_feature_vector[df_feature_vector["risk_score"] == "Critical"].copy()

    # Sort by acq_date (and acq_time if available) descending
    sort_cols = ["acq_date"]
    if "acq_time" in critical_df.columns:
        critical_df["_sort_time"] = pd.to_numeric(critical_df["acq_time"], errors="coerce").fillna(0)
        sort_cols.append("_sort_time")

    critical_df = critical_df.sort_values(by=sort_cols, ascending=False)
    if "_sort_time" in critical_df.columns:
        critical_df.drop(columns=["_sort_time"], inplace=True)

    alerts_list = []
    for idx, row in critical_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        frp = float(row["frp"])
        hist_mean = float(row["historical_mean_frp"]) if pd.notnull(row["historical_mean_frp"]) else None
        dev = float(row["frp_deviation"]) if pd.notnull(row["frp_deviation"]) else None
        dist_res = float(row["distance_to_residential_m"]) if pd.notnull(row["distance_to_residential_m"]) else None
        esc = str(row["escalation_status"]) if "escalation_status" in row and pd.notnull(row["escalation_status"]) else "Abnormal"
        alert_flag = str(row["alert_flag"]) if pd.notnull(row["alert_flag"]) else "Abnormal spike"
        cid = int(row["cluster_id"]) if pd.notnull(row["cluster_id"]) else -1
        acq_date = str(row["acq_date"])

        # Construct actionable, clear narrative
        dev_str = f"{dev:.1f}x baseline" if dev is not None else "elevated"
        mean_str = f"{hist_mean:.1f} MW" if hist_mean is not None else "baseline"
        dist_str = f"{dist_res:,.0f} m" if dist_res is not None else "close proximity"

        narrative = (
            f"Critical hazard at ({lat:.4f}, {lon:.4f}): Thermal spike of {frp:.1f} MW "
            f"({dev_str}, normal avg {mean_str}) detected {dist_str} from residential settlement. "
            f"Status: {esc}."
        )

        alerts_list.append({
            "row_index": int(idx),
            "cluster_id": cid,
            "latitude": lat,
            "longitude": lon,
            "acq_date": acq_date,
            "frp": frp,
            "historical_mean_frp": hist_mean,
            "frp_deviation": dev,
            "distance_to_residential_m": dist_res,
            "escalation_status": esc,
            "alert_flag": alert_flag,
            "risk_score": "Critical",
            "class_label": row.get("class_label", "Industrial"),
            "ml_predicted_label": row.get("ml_predicted_label", row.get("class_label", "Industrial")),
            "narrative": narrative
        })

    return {
        "count": len(alerts_list),
        "alerts": alerts_list
    }


@app.post("/predict/existing")
def predict_existing(req: ExistingPredictionRequest):
    """
    3. POST /predict/existing
    Accepts row index or lat/lon, looks up precomputed feature vector, runs classifier live,
    and returns predicted class, confidence, and top 3 SHAP contributing features.
    """
    row_idx = req.row_index

    # If coordinates provided instead of row_index, find nearest row
    if row_idx is None:
        if req.latitude is None or req.longitude is None:
            raise HTTPException(status_code=400, detail="Must provide either row_index or latitude and longitude.")

        # Find closest row by Euclidean distance
        dists = np.sqrt(
            (df_feature_vector["latitude"] - req.latitude) ** 2 +
            (df_feature_vector["longitude"] - req.longitude) ** 2
        )
        row_idx = int(dists.idxmin())

    if row_idx < 0 or row_idx >= len(df_feature_vector):
        raise HTTPException(status_code=404, detail=f"Row index {row_idx} out of range (0 - {len(df_feature_vector)-1}).")

    target_row = df_feature_vector.iloc[[row_idx]]
    X_mat = build_model_matrix(target_row)

    res = compute_prediction_and_shap(X_mat, target_index=0)

    # Attach point metadata for display
    res["row_index"] = row_idx
    res["latitude"] = float(target_row["latitude"].iloc[0])
    res["longitude"] = float(target_row["longitude"].iloc[0])
    res["acq_date"] = str(target_row["acq_date"].iloc[0]) if "acq_date" in target_row.columns else "N/A"
    res["frp"] = float(target_row["frp"].iloc[0])
    res["rule_label"] = str(target_row["class_label"].iloc[0]) if "class_label" in target_row.columns else "N/A"
    res["risk_score"] = str(target_row["risk_score"].iloc[0]) if "risk_score" in target_row.columns else "N/A"
    res["escalation_status"] = str(target_row["escalation_status"].iloc[0]) if "escalation_status" in target_row.columns else "N/A"

    return res


@app.post("/predict/hypothetical")
def predict_hypothetical(req: HypotheticalPredictionRequest):
    """
    4. POST /predict/hypothetical
    Accepts arbitrary latitude, longitude, and frp value.
    Computes distance to cached OSM industrial and farmland polygons using local geometry math.
    Marks all missing features (historical stats, satellite indices) explicitly as missing,
    applies median imputation, runs model, and returns predicted class with top 3 SHAP drivers.
    """
    lat = req.latitude
    lon = req.longitude
    frp = req.frp

    # Local geometry distance calculation in UTM Zone 44N (EPSG:32644)
    pt_utm = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=32644).iloc[0]
    dist_ind_m = float(pt_utm.distance(industrial_union))
    dist_farm_m = float(pt_utm.distance(farmland_union))

    # Assemble raw feature record for hypothetical point
    hypo_record = {
        "latitude": lat,
        "longitude": lon,
        "frp": frp,
        "daynight": req.daynight,
        "confidence": req.confidence,
        "brightness": TRAINING_MEDIANS.get("brightness", 335.0),
        "distance_to_industrial_m": dist_ind_m,
        "distance_to_farmland_m": dist_farm_m,
        "distinct_days": 1.0,  # Single new detection
        # Historical cluster stats are missing for new hypothetical points
        "historical_mean_frp": np.nan,
        "historical_max_frp": np.nan,
        "historical_std_frp": np.nan,
        "frp_deviation": np.nan,
        "night_detection_ratio": np.nan,
        # Sentinel-2 indices are missing for unobserved coordinates
        "ndvi": np.nan,
        "ndbi": np.nan,
        "ndwi": np.nan,
    }

    df_hypo = pd.DataFrame([hypo_record])
    X_mat = build_model_matrix(df_hypo)

    res = compute_prediction_and_shap(X_mat, target_index=0)

    # Attach hypothetical point metadata
    res["latitude"] = lat
    res["longitude"] = lon
    res["frp"] = frp
    res["distance_to_industrial_m"] = round(dist_ind_m, 1)
    res["distance_to_farmland_m"] = round(dist_farm_m, 1)
    res["mode"] = "hypothetical"

    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
