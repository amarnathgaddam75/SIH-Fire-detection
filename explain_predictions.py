"""
explain_predictions.py
======================
Model Interpretability via SHAP (SHapley Additive exPlanations)
for the GeoFlare XGBoost classifier (data/classifier.pkl).

Pipeline:
1. Loads classifier.pkl and data/feature_vector.csv with identical preprocessing.
2. Creates shap.TreeExplainer and computes SHAP values on the held-out test set.
3. Generates and saves a global summary bar chart to data/shap_summary.png.
4. Explains individual predictions for Industrial, Natural Fire, and Unknown classes
   in plain, human-readable language showing top 5 drivers.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/WSL
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Setup Paths
# --------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "classifier.pkl")
CSV_PATH = os.path.join(DATA_DIR, "feature_vector.csv")
CHART_PATH = os.path.join(DATA_DIR, "shap_summary.png")

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


def load_and_preprocess_data():
    """Load feature_vector.csv and perform exact preprocessing from train_model.py."""
    df = pd.read_csv(CSV_PATH)

    # Encode daynight and confidence
    df["daynight"] = df["daynight"].map({"D": 1.0, "N": 0.0}).astype(float)
    df["confidence"] = df["confidence"].map({"l": 0.0, "n": 1.0, "h": 2.0}).astype(float)

    # Clean numeric indices
    for col in ["ndvi", "ndbi", "ndwi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df["class_label"].copy()

    # Impute missing values & add flags
    cols_with_missing = X.columns[X.isnull().any()].tolist()
    for col in cols_with_missing:
        X[f"{col}_was_missing"] = X[col].isnull().astype(int)
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X[col] = X[col].fillna(median_val)

    # Stratified split matching train_model.py exactly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Retrieve original metadata for the test set for context (lat/lon/acq_date)
    test_indices = X_test.index
    meta_test = df.loc[test_indices, ["latitude", "longitude", "acq_date"]].copy()

    return X_train, X_test, y_train, y_test, meta_test


def format_feature_value(feat_name, val):
    """Format numeric values with intuitive units."""
    if "distance" in feat_name:
        return f"{val:,.1f}m"
    elif feat_name in ["frp", "historical_mean_frp", "historical_max_frp", "historical_std_frp"]:
        return f"{val:.1f} MW"
    elif feat_name == "brightness":
        return f"{val:.1f} K"
    elif feat_name == "distinct_days":
        return f"{int(val)} days"
    elif feat_name == "night_detection_ratio":
        return f"{val * 100:.1f}% night"
    elif feat_name == "daynight":
        return "Day (1)" if val == 1.0 else "Night (0)"
    elif feat_name == "confidence":
        mapping = {0.0: "low (0)", 1.0: "nominal (1)", 2.0: "high (2)"}
        return mapping.get(val, f"{val:.0f}")
    elif "_was_missing" in feat_name:
        return "Yes (1)" if val == 1.0 else "No (0)"
    elif feat_name == "frp_deviation":
        return f"{val:.2f}x baseline"
    else:
        return f"{val:.2f}"


def explain_single_sample(row_idx, class_target, X_test, y_test, meta_test, preds, probs, shap_values, class_names):
    """Print top 5 SHAP drivers for a specific test sample."""
    sample_feat = X_test.iloc[row_idx]
    actual_label = y_test.iloc[row_idx]
    pred_label = preds[row_idx]
    pred_probs = probs[row_idx]
    meta = meta_test.iloc[row_idx]
    target_class_idx = class_names.index(class_target)

    # Extract SHAP contributions for the target predicted class
    # shap_values has shape (N_samples, N_features, N_classes)
    sample_shap = shap_values[row_idx, :, target_class_idx]

    # Rank features by absolute magnitude of SHAP value
    top_indices = np.argsort(np.abs(sample_shap))[::-1][:5]

    print("=" * 80)
    print(f"EXAMPLE PREDICTION: {class_target.upper()}")
    print("=" * 80)
    print(f"  Test Sample Index:    {row_idx} (Original Row ID: {sample_feat.name})")
    print(f"  Acquisition Date:     {meta['acq_date']} | Location: ({meta['latitude']:.5f}, {meta['longitude']:.5f})")
    print(f"  Ground Truth Class:   {actual_label}")
    print(f"  Model Predicted Class: {pred_label} (Confidence: {pred_probs[target_class_idx]*100:.1f}%)")
    print("  Class Probabilities:")
    for c_name, p in zip(class_names, pred_probs):
        print(f"    - {c_name:22s}: {p*100:5.1f}%")

    print(f"\n  Top 5 Features Driving Prediction Toward '{class_target}':")
    print("  " + "-" * 76)
    for rank, feat_idx in enumerate(top_indices, 1):
        feat_name = X_test.columns[feat_idx]
        val = sample_feat.iloc[feat_idx]
        shap_val = sample_shap[feat_idx]
        val_str = format_feature_value(feat_name, val)

        if shap_val > 0:
            direction = f"pushed strongly TOWARD {class_target} (+{shap_val:.3f})"
        else:
            direction = f"pushed AWAY from {class_target} ({shap_val:.3f})"

        print(f"    {rank}. {feat_name} = {val_str:<18} --> {direction}")
    print()


def main():
    print("=" * 80)
    print("GeoFlare -- SHAP Model Explainability Pipeline")
    print("=" * 80)

    # 1. Load Model & Data
    print("\n[Part 1] Loading trained model and feature vector...")
    model = joblib.load(MODEL_PATH)
    print(f"  Loaded model: {type(model).__name__} from {MODEL_PATH}")

    X_train, X_test, y_train, y_test, meta_test = load_and_preprocess_data()
    print(f"  Test set loaded: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_test]))
    class_names = list(le.classes_)
    print(f"  Target classes ({len(class_names)}): {class_names}")

    # Generate predictions and probabilities
    preds_encoded = model.predict(X_test)
    preds = le.inverse_transform(preds_encoded)
    probs = model.predict_proba(X_test)

    # 2. TreeExplainer & Compute SHAP Values
    print("\n[Part 2] Computing SHAP values using shap.TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # shap_values shape: (N_samples, N_features, N_classes)
    shap_vals = explainer.shap_values(X_test)
    shap_vals = np.array(shap_vals)
    print(f"  Computed SHAP matrix: {shap_vals.shape} (test_samples, features, classes)")

    # 3. Global Summary Bar Chart
    print("\n[Part 3] Generating global feature importance summary bar chart...")
    # Compute mean absolute SHAP value across all samples and classes
    # Shape: (N_features,)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 2))
    feat_importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=True)

    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    plt.barh(feat_importance_df["feature"], feat_importance_df["mean_abs_shap"], color="#1f77b4", edgecolor="#0e436b")
    plt.xlabel("Mean |SHAP Value| (Average impact across all classes)", fontsize=11, fontweight="bold")
    plt.title("Global Feature Importance (SHAP TreeExplainer on Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(CHART_PATH)
    plt.close()
    print(f"  Global summary bar chart saved to: {CHART_PATH}")

    # 4. Individual Case Explanations
    print("\n[Part 4] Individual Prediction Explanations (Industrial, Natural Fire, Unknown)...")

    # Pick representative examples correctly predicted with high certainty or typical attributes
    classes_to_explain = ["Industrial", "Natural Fire", "Unknown"]

    for c in classes_to_explain:
        # Filter test rows where prediction matches target class
        matching_indices = np.where(preds == c)[0]
        if len(matching_indices) == 0:
            print(f"  Warning: No test rows predicted as {c}")
            continue

        # Choose the first matching row or highest probability row
        c_idx = class_names.index(c)
        best_idx = matching_indices[np.argmax(probs[matching_indices, c_idx])]
        explain_single_sample(best_idx, c, X_test, y_test, meta_test, preds, probs, shap_vals, class_names)

    print("=" * 80)
    print("Explainability pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
