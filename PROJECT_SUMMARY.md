# GeoFlare (SIH26162) — Complete Project Technical Summary

**Problem Statement**: SIH26162 — AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources Using NASA FIRMS, OpenStreetMap (OSM) & Satellite Data  
**Repository**: `git@github.com:amarnathgaddam75/SIH-Fire-detection.git` (Branch: `main`)  
**Timestamp**: September 8, 2026  

---

## 1. Executive Overview

GeoFlare is an end-to-end intelligence system that separates ephemeral wildland and crop fires from permanent industrial thermal sources (flare stacks, smelters, coal washeries, cement kilns, and power plants). 

The platform blends:
1. **Multi-sensor satellite thermal hotspots** from NASA FIRMS (VIIRS & MODIS).
2. **OpenStreetMap (OSM) spatial topology** (industrial, agricultural, and residential polygon distances).
3. **Spatio-temporal persistence clustering** (DBSCAN at $500\,\text{m}$ metric radius).
4. **Machine Learning Classifiers** (XGBoost and Random Forest with 5-fold cross-validation).
5. **Model Interpretability** (SHAP Shapley Additive exPlanations).
6. **Optical Satellite Validation** (Copernicus Data Space Ecosystem / Sentinel Hub Statistical API for Sentinel-2 L2A surface reflectance indices: NDVI, NDBI, NDWI).
7. **Population Hazard Assessment** (explainable point-based risk scoring incorporating residential proximity).
8. **Interactive Geospatial Dashboard** (Leaflet map with dynamic filtering, disagreement flags, and side-by-side verification popups).

---

## 2. Study Areas & Geographical Context

### Primary Focus Region: Korba-Raigarh Industrial Belt (Chhattisgarh)
- **Bounding Box**: North `22.6`, South `22.1`, East `83.1`, West `82.3`
- **Projection**: `EPSG:32644` (UTM Zone 44N)
- **Characteristics**: Coal mining basin, thermal power generation, metal smelters, and dense industrial infrastructure intermingled with residential colonies and Sal forests.
- **Dataset**: 2,008 hotspot detections.

### Cross-Region Validation: Punjab Stubble-Burning Belt
- **Bounding Box**: North `30.6`, South `29.6`, East `76.5`, West `75.0`
- **Projection**: `EPSG:32643` (UTM Zone 43N)
- **Characteristics**: Heavy agricultural plains with seasonal post-harvest paddy stubble burning; absence of permanent flare stacks.
- **Dataset**: 1,350 hotspot detections.

---

## 3. Core Technical Pipeline & File Inventory

```
SIH/
├── classify_hotspots.py            # Initial spatial join & DBSCAN persistence clustering (Korba)
├── build_feature_table.py          # 23-feature engineering, OSM distance calc, 5-class rules (Korba)
├── train_model.py                  # RF & XGBoost training, 5-fold CV, model evaluation & serialization
├── explain_predictions.py          # SHAP TreeExplainer, global bar chart, plain-language driver summaries
├── predict.py                      # Reusable inference engine (single-row & batch CLI)
├── add_ml_predictions.py           # Evaluates model, appends ml_predicted_label, exports firms_final_ml.geojson
├── export_unknown_rows.py          # Extracts borderline Unknown rows with failure condition diagnostics
├── reclassify_ag_burn.py           # Calibrated agricultural thresholding (distance_to_farmland <= 2000m)
├── compute_residential_risk.py     # OSM residential distance & explainable point-based risk scoring
├── fetch_sentinel_pilot.py         # CDSE Sentinel Hub Statistical API ingestion (NDVI/NDBI/NDWI)
├── classify_hotspots_punjab.py     # Punjab regional classifier pipeline
├── build_feature_table_punjab.py   # Punjab feature engineering pipeline
├── train_model_punjab.py           # Punjab training & Korba cross-region zero-shot transfer test
├── dashboard.html                  # Interactive Leaflet dashboard with AI disagreement visual cues
├── results_summary.md              # Formal technical audit report of training metrics
├── data/
│   ├── classifier.pkl              # Production XGBoost model (trained on Korba 23-feature schema)
│   ├── model_features.txt          # Exact 23-feature ordered schema
│   ├── feature_vector.csv          # 2,008 rows enriched with distances, ML predictions, risk scores
│   ├── firms_final_ml.geojson      # GeoJSON formatted for dashboard with ML properties & disagreement flags
│   ├── firms_final.geojson         # Base GeoJSON synced with risk properties
│   ├── shap_summary.png            # High-resolution global SHAP feature importance bar chart
│   ├── unknown_rows_review.csv     # 74 unclassified / borderline events for human-in-the-loop review
│   └── sentinel2_pilot_results.csv # 19 pilot targets with real CDSE Sentinel-2 indices
├── data_punjab/                    # Independent dataset, features, and model for Punjab
│   ├── feature_vector_punjab.csv
│   ├── firms_labeled_punjab.csv
│   └── classifier_punjab.pkl
└── cache/                          # Cached GeoJSON boundaries (industrial, farmland, residential)
```

---

## 4. Key Milestones & Mathematical Details

### Milestone 1: 5-Class Typology & Feature Engineering
Rather than binary classification, GeoFlare established a 5-class typology:
1. **`Industrial`**: $\text{distance\_to\_industrial\_m} \le 1000\,\text{m}$ AND ($\text{type} == 2$ OR $\text{distinct\_days} \ge 10$).
2. **`Agricultural Burning`**: $\text{distance\_to\_farmland\_m} \le 2000\,\text{m}$ AND $\text{distance\_to\_industrial\_m} > 1000\,\text{m}$ AND $\text{distinct\_days} \le 3$.
3. **`Natural Fire`**: NASA $\text{type} == 0$ AND $\text{distance\_to\_industrial\_m} > 1000\,\text{m}$.
4. **`Unknown`**: Ambiguous, conflicting, or borderline detections (e.g. fire inside industrial polygon with $<10$ days persistence).

- **23-Feature Matrix**: Base attributes (`frp`, `brightness`, `confidence`, `daynight`, `distance_to_industrial_m`, `distance_to_farmland_m`, `distinct_days`, `historical_mean_frp`, `historical_max_frp`, `historical_std_frp`, `frp_deviation`, `night_detection_ratio`, `ndvi`, `ndbi`, `ndwi`) plus 8 companion `*_was_missing` binary flags with median imputation.

### Milestone 2: Machine Learning Model Evaluation
Evaluated on an 80/20 stratified split (1,606 train / 402 test):
- **5-Fold Stratified Cross-Validation (Macro F1)**: Mean = **0.9149** (Std = 0.0630).
- **Random Forest**: Test Accuracy: 98.76%, Test Macro F1: **0.9108**, Unknown Recall: 80.00%.
- **XGBoost (Winner — `data/classifier.pkl`)**:
  - Test Accuracy: **99.25%**
  - Test Macro F1: **0.9721**
  - Per-Class Precision / Recall:
    - `Agricultural Burning`: 100.0% / 100.0%
    - `Industrial`: 100.0% / 99.53%
    - `Natural Fire`: 98.86% / 100.0%
    - `Unknown`: 92.86% / **86.67%** (correctly isolated 13 of 15 ambiguous events)

### Milestone 3: Model Explainability via SHAP
- Formulated with `shap.TreeExplainer` producing a 3D tensor $(402 \times 23 \times 4)$.
- **Global Importance Hierarchy**:
  1. `distance_to_farmland_m` (25.01%)
  2. `distance_to_industrial_m` (20.02%)
  3. `distinct_days` (16.56%)
  4. `historical_max_frp` (7.92%)
  5. `night_detection_ratio` (5.28%)
- **Plain-Language Case Findings**:
  - *Industrial*: Driven by high persistence (`distinct_days = 98 days`, $+5.087$ SHAP log-odds) and night operation (`92.3% night`, $+0.580$).
  - *Natural Fire*: Driven by spatial remoteness (`dist_industrial = 7.77 km`, $+2.112$; `dist_farmland = 20.17 km`, $+2.044$).
  - *Unknown*: Driven by spatial proximity to industrial zones ($239.4\,\text{m}$) conflicting with zero persistence and low brightness.

### Milestone 4: Production Inference & Dashboard Integration
- [`predict.py`](file:///Ubuntu/home/dev_amar75/SIH/predict.py) was built as an autonomous CLI and API predictor.
- [`add_ml_predictions.py`](file:///Ubuntu/home/dev_amar75/SIH/add_ml_predictions.py) scored all 2,008 hotspots, identifying **85 ML-vs-Rule disagreements (4.23%)**:
  - **61 points**: Previously forced into Industrial by heuristic rules, but the ML model identifies them as `Unknown` due to low persistence.
  - **12 points**: Previously marked Natural Fire, resolved to `Agricultural Burning`.
  - **12 points**: Previously marked Natural Fire, resolved to `Unknown`.
- [`dashboard.html`](file:///Ubuntu/home/dev_amar75/SIH/dashboard.html) was upgraded to render `firms_final_ml.geojson`:
  - Visual Distinction: **Dashed border** (`#facc15`, `dashArray: '4, 4'`, `weight: 3.5`) for all disagreement points.
  - Popups: Side-by-side comparison cards (`Rule-Based` vs. `ML XGBoost` with confidence %).
  - Filter Controls: "Show Disagreements Only" checkbox isolating the 85 points of interest.

### Milestone 5: Cross-Region Transfer to Punjab
- Tested the Korba-trained model zero-shot on 1,350 unseen Punjab hotspot detections:
  - **Overall Accuracy**: **98.74%** | **Macro F1**: **0.9392**
  - **Agricultural Burning Recall**: **98.78% (81 / 82 detected)**
  - **Zero False Industrial Classifications**: Predicted 0 industrial points in Punjab, confirming the model does not confuse crop fires with factories.

### Milestone 6: Real Sentinel-2 L2A Ingestion via CDSE
- Configured persistent OAuth2 profile `cdse` in `~/.config/sentinelhub/config.toml`.
- Queried Sentinel Hub Statistical API on CDSE for 19 pilot points ($500\,\text{m} \times 500\,\text{m}$ bounding boxes at $10\,\text{m}$ resolution, 30-day aggregation window, Sentinel-2 L2A).
- **100% Query Success**: Real mean `ndvi`, `ndbi`, `ndwi` calculated and written to [`data/feature_vector.csv`](file:///Ubuntu/home/dev_amar75/SIH/data/feature_vector.csv).
  - *Physical Validation*: Industrial sites demonstrated high NDBI ($+0.24$ to $+0.36$) and suppressed NDVI ($+0.04$ to $+0.08$), whereas wildland baselines exhibited higher NDVI ($+0.19$ to $+0.26$).

### Milestone 7: Residential Proximity & Multi-Factor Risk Scoring
- Queried 81 OSM residential settlements in Korba; computed `distance_to_residential_m`.
- Point-based explainable risk score:
  $$\text{Total Points} = \text{Class Hazard (0-3)} + \text{Spike Hazard (0-3)} + \text{Proximity Hazard (0-3)}$$
- **Distribution**:
  - `Critical` ($\ge 7$ pts): **31 rows (1.5%)** — severe thermal surges ($2.5\times - 6.2\times$ baseline) within $2.5\,\text{km}$ of homes (16 within $500\,\text{m}$).
  - `High` ($5-6$ pts): **586 rows (29.2%)** — steady industrial heat near communities.
  - `Medium` ($3-4$ pts): **503 rows (25.0%)** — moderate distance thermal sources.
  - `Low` ($0-2$ pts): **888 rows (44.2%)** — remote wildland fires.
- Synced across `data/feature_vector.csv`, `data/firms_final_ml.geojson`, and `data/firms_final.geojson`.

---

## 5. Summary Statistics Table

| Metric / Dimension | Korba Primary Dataset | Punjab Validation Dataset |
| :--- | :---: | :---: |
| **Total Hotspot Detections** | **2,008** | **1,350** |
| **UTM Zone & CRS** | Zone 44N (`EPSG:32644`) | Zone 43N (`EPSG:32643`) |
| **Industrial Class Count** | 1,054 (52.5%) | 0 (0.0%) |
| **Natural Fire Class Count** | 868 (43.2%) | 1,168 (86.5%) |
| **Agricultural Burning Count** | 12 (0.6%) | **82 (6.1%)** |
| **Unknown Class Count** | 74 (3.7%) | 100 (7.4%) |
| **XGBoost Test Macro F1** | **0.9721** | **0.9392** *(zero-shot transfer)* |
| **ML-vs-Rule Disagreements** | 85 (4.23%) | — |
| **Critical Hazard Hotspots** | 31 (1.5%) | — |
| **CDSE Sentinel-2 Pilot Target Success** | 19 / 19 (100.0%) | — |

---

## 6. How to Run Any Part of the System

```bash
# 1. Run inference on a specific row or entire batch:
python3 predict.py --row 10
python3 predict.py --batch

# 2. Re-compute ML predictions and update GeoJSON:
python3 add_ml_predictions.py

# 3. Re-run model interpretability and save SHAP plot:
python3 explain_predictions.py

# 4. Re-calculate residential distances and risk scoring:
python3 compute_residential_risk.py

# 5. Query CDSE Sentinel-2 Statistical API:
python3 fetch_sentinel_pilot.py

# 6. Execute Punjab stubble-burning pipeline:
python3 classify_hotspots_punjab.py
python3 build_feature_table_punjab.py
python3 train_model_punjab.py

# 7. View Dashboard locally:
# Open dashboard.html in any modern browser (or serve with python3 -m http.server 8000)
```
