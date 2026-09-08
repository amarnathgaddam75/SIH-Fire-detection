"""
predict.py  --  GeoFlare Classifier Inference Engine
====================================================
Loads the trained model (data/classifier.pkl), feature schema (data/model_features.txt),
and can classify any row from data/feature_vector.csv or custom input dictionaries,
returning the predicted class_label and prediction probabilities.

Usage:
    python3 predict.py                    # Runs demo inference on sample rows
    python3 predict.py --row 10           # Classifies row 10 of feature_vector.csv
    python3 predict.py --batch            # Classifies all rows and prints summary
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "classifier.pkl")
FEATURES_PATH = os.path.join(DATA_DIR, "model_features.txt")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")

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


class GeoFlarePredictor:
    def __init__(self, model_path=MODEL_PATH, features_path=FEATURES_PATH, training_csv=CSV_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Feature list not found at {features_path}")

        self.model = joblib.load(model_path)

        with open(features_path, "r") as f:
            self.expected_features = [line.strip() for line in f if line.strip()]

        # Load reference training data for consistent label mapping and median imputation
        self.df_ref = pd.read_csv(training_csv)
        self.le = LabelEncoder()
        self.le.fit(self.df_ref["class_label"])
        self.classes = list(self.le.classes_)

        # Compute training medians for imputation
        self.medians = {}
        for col in FEATURE_COLUMNS:
            series = pd.to_numeric(self.df_ref[col], errors="coerce")
            med = series.median()
            self.medians[col] = 0.0 if pd.isna(med) else med

    def preprocess_dataframe(self, df_input):
        """Transform raw input DataFrame into the exact 23-feature matrix."""
        df = df_input.copy()

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

        # Missing indicator flags and median filling
        X = pd.DataFrame(index=df.index)
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
            else:
                series = pd.Series(np.nan, index=df.index)

            # Companion was_missing flag
            flag_col = f"{col}_was_missing"
            if flag_col in self.expected_features:
                X[flag_col] = series.isnull().astype(int)

            # Impute median
            X[col] = series.fillna(self.medians.get(col, 0.0))

        # Reorder to match model_features.txt precisely
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.expected_features]
        return X

    def predict_row(self, row_dict_or_series):
        """Classify a single row and return (predicted_label, confidence, prob_dict)."""
        if isinstance(row_dict_or_series, pd.Series):
            df_single = row_dict_or_series.to_frame().T
        elif isinstance(row_dict_or_series, dict):
            df_single = pd.DataFrame([row_dict_or_series])
        else:
            df_single = pd.DataFrame(row_dict_or_series)

        X = self.preprocess_dataframe(df_single)
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        pred_label = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {c: float(probs[i]) for i, c in enumerate(self.classes)}
        return pred_label, confidence, prob_dict

    def predict_dataframe(self, df_input):
        """Classify a full DataFrame, returning array of labels and array of confidences."""
        X = self.preprocess_dataframe(df_input)
        probs = self.model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = self.le.inverse_transform(pred_indices)
        confidences = np.max(probs, axis=1)
        return pred_labels, confidences, probs


def main():
    parser = argparse.ArgumentParser(description="GeoFlare XGBoost Model Inference")
    parser.add_argument("--row", type=int, default=None, help="Classify a specific row index from feature_vector.csv")
    parser.add_argument("--batch", action="store_true", help="Classify entire feature_vector.csv and show summary")
    args = parser.parse_args()

    predictor = GeoFlarePredictor()
    df_all = pd.read_csv(CSV_PATH)
    print("=" * 75)
    print(f"GeoFlare Model Predictor -- Loaded {len(df_all):,} rows from {os.path.basename(CSV_PATH)}")
    print(f"Classes: {predictor.classes}")
    print("=" * 75)

    if args.row is not None:
        idx = args.row
        if idx < 0 or idx >= len(df_all):
            print(f"Error: row index {idx} out of range (0 to {len(df_all)-1})")
            return
        row = df_all.iloc[idx]
        pred_label, conf, probs = predictor.predict_row(row)
        true_label = row.get("class_label", "N/A")
        print(f"\nRow {idx}:")
        print(f"  Coordinates: ({row.get('latitude', 'N/A')}, {row.get('longitude', 'N/A')}) | Date: {row.get('acq_date', 'N/A')}")
        print(f"  Ground Truth:    {true_label}")
        print(f"  Predicted Label: {pred_label} (Confidence: {conf*100:.2f}%)")
        print(f"  Probabilities:   {probs}")
        return

    if args.batch:
        print("\nRunning batch classification across all rows...")
        labels, confs, probs = predictor.predict_dataframe(df_all)
        df_all["ml_pred"] = labels
        df_all["ml_conf"] = confs
        print("\nPrediction Breakdown:")
        for label, count in df_all["ml_pred"].value_counts().items():
            pct = count / len(df_all) * 100
            print(f"  {label:25s}: {count:5,d} ({pct:5.1f}%)")
        agreed = (df_all["class_label"] == df_all["ml_pred"]).sum()
        print(f"\nAgreement with rule-based class_label: {agreed:,} / {len(df_all):,} ({agreed/len(df_all)*100:.2f}%)")
        return

    # Default: Demo mode on representative samples of each class
    print("\nDEMO: Testing classification on representative samples of each class:")
    print("-" * 75)
    for target_class in predictor.classes:
        matching = df_all[df_all["class_label"] == target_class]
        if len(matching) > 0:
            sample_row = matching.iloc[0]
            idx = sample_row.name
            pred_label, conf, probs = predictor.predict_row(sample_row)
            print(f"[{target_class.upper()} Sample - Row {idx}]")
            print(f"  Lat/Lon: ({sample_row['latitude']:.4f}, {sample_row['longitude']:.4f}) | FRP: {sample_row['frp']} MW")
            print(f"  True Label:      {target_class}")
            print(f"  Predicted Label: {pred_label} (Confidence: {conf*100:.2f}%)")
            prob_str = ", ".join([f"{c}: {p*100:.1f}%" for c, p in probs.items()])
            print(f"  Probabilities:   [{prob_str}]")
            print("-" * 75)


if __name__ == "__main__":
    main()
