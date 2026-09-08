"""
add_ml_predictions.py
=====================
Runs the trained GeoFlare classifier (data/classifier.pkl) against every row in
data/feature_vector.csv and adds two new columns:
  - ml_predicted_label
  - ml_confidence

Merges these predictions into a new GeoJSON: data/firms_final_ml.geojson,
so each feature's properties contain both the original rule-based final_label
and the new ml_predicted_label, as well as a disagreement indicator.
"""

import os
import json
import numpy as np
import pandas as pd
from predict import GeoFlarePredictor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
INPUT_GEOJSON = os.path.join(DATA_DIR, "firms_final.geojson")
OUTPUT_GEOJSON = os.path.join(DATA_DIR, "firms_final_ml.geojson")


def main():
    print("=" * 75)
    print("GeoFlare -- Generating ML Predictions for Dashboard Integration")
    print("=" * 75)

    if not os.path.exists(INPUT_GEOJSON):
        raise FileNotFoundError(f"Input GeoJSON not found: {INPUT_GEOJSON}")

    # 1. Load data and predictor
    print(f"\n1. Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"   Loaded {len(df):,} rows.")

    predictor = GeoFlarePredictor()
    print(f"   Predictor ready with classes: {predictor.classes}")

    # 2. Run predictions across all rows
    print("\n2. Computing model predictions and confidence scores...")
    pred_labels, confidences, probs = predictor.predict_dataframe(df)

    df["ml_predicted_label"] = pred_labels
    df["ml_confidence"] = np.round(confidences, 4)

    # Save back to feature_vector.csv as well
    df.to_csv(CSV_PATH, index=False)
    print(f"   Updated {CSV_PATH} with ml_predicted_label and ml_confidence columns.")

    # 3. Load input GeoJSON
    print(f"\n3. Loading {INPUT_GEOJSON}...")
    with open(INPUT_GEOJSON, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    print(f"   GeoJSON contains {len(features):,} features.")

    if len(features) != len(df):
        print(f"   Warning: GeoJSON feature count ({len(features)}) != CSV row count ({len(df)})")

    # 4. Merge predictions into GeoJSON properties
    print("\n4. Merging ML predictions into GeoJSON properties...")
    disagreements = 0
    disagreement_types = {}

    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        row = df.iloc[i]

        ml_label = str(row["ml_predicted_label"])
        ml_conf = float(row["ml_confidence"])
        rule_label = str(props.get("final_label", row.get("class_label", "Unknown")))

        # Check disagreement between rule-based final_label and ML prediction
        is_disagreement = (ml_label != rule_label)
        if is_disagreement:
            disagreements += 1
            key = f"{rule_label} -> {ml_label}"
            disagreement_types[key] = disagreement_types.get(key, 0) + 1

        # Add ML attributes
        props["ml_predicted_label"] = ml_label
        props["ml_confidence"] = ml_conf
        props["ml_disagreement"] = is_disagreement
        props["class_label"] = row.get("class_label", rule_label)

        # Store probability breakdown for popups
        row_probs = {c: round(float(probs[i, c_idx]), 3) for c_idx, c in enumerate(predictor.classes)}
        props["ml_probs"] = row_probs

        feat["properties"] = props

    # 5. Write to firms_final_ml.geojson
    print(f"\n5. Saving merged GeoJSON to {OUTPUT_GEOJSON}...")
    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    file_size_kb = os.path.getsize(OUTPUT_GEOJSON) / 1024
    print(f"   Successfully wrote {len(features):,} features to {OUTPUT_GEOJSON} ({file_size_kb:.1f} KB)")

    # 6. Summary of agreement vs disagreement
    print("\n" + "=" * 75)
    print("PREDICTION & AGREEMENT SUMMARY:")
    print("=" * 75)
    print(f"  Total Hotspots:                 {len(features):,}")
    print(f"  Agreement (Rule == ML):         {len(features) - disagreements:,} ({(len(features) - disagreements)/len(features)*100:.2f}%)")
    print(f"  Disagreements (Rule != ML):     {disagreements:,} ({disagreements/len(features)*100:.2f}%)")
    print("\n  Disagreement Breakdown (Rule final_label -> ML predicted):")
    for trans, count in sorted(disagreement_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {trans:30s}: {count:4d} points")
    print("=" * 75)


if __name__ == "__main__":
    main()
