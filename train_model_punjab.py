"""
train_model_punjab.py
=====================
Trains and evaluates models on the Punjab feature table, and evaluates the
existing Korba-trained classifier (data/classifier.pkl) directly on Punjab data.

Sections:
1. Load data_punjab/feature_vector_punjab.csv & inspect class distribution.
2. Side-by-side comparison of class_label distribution: Korba vs Punjab.
3. Feature engineering & missing value imputation (23 features).
4. Train/test split (80/20 stratified) & train Punjab model.
5. Cross-Region Evaluation: evaluate Korba-trained model (data/classifier.pkl)
   against the Punjab dataset (Accuracy, Classification Report, Confusion Matrix).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_KORBA = os.path.join(SCRIPT_DIR, "data")
DATA_DIR_PUNJAB = os.path.join(SCRIPT_DIR, "data_punjab")

KORBA_CSV = os.path.join(DATA_DIR_KORBA, "feature_vector.csv")
KORBA_MODEL = os.path.join(DATA_DIR_KORBA, "classifier.pkl")
KORBA_FEATURES = os.path.join(DATA_DIR_KORBA, "model_features.txt")

PUNJAB_CSV = os.path.join(DATA_DIR_PUNJAB, "feature_vector_punjab.csv")
PUNJAB_MODEL = os.path.join(DATA_DIR_PUNJAB, "classifier_punjab.pkl")

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


def load_and_preprocess(csv_path, expected_feature_list=None):
    """Load and preprocess feature vector into 23 features."""
    df = pd.read_csv(csv_path)

    # Ordinal / binary mappings
    df["daynight"] = df["daynight"].map({"D": 1.0, "N": 0.0}).astype(float)
    df["confidence"] = df["confidence"].map({"l": 0.0, "n": 1.0, "h": 2.0}).astype(float)

    for col in ["ndvi", "ndbi", "ndwi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[FEATURE_COLUMNS].copy()
    y = df["class_label"].copy()

    # Missing flags & median imputation
    cols_with_missing = X.columns[X.isnull().any()].tolist()
    for col in cols_with_missing:
        X[f"{col}_was_missing"] = X[col].isnull().astype(int)
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        X[col] = X[col].fillna(med)

    # Ensure all expected features are present (in case a region had 0 missing for a col)
    if expected_feature_list:
        for f in expected_feature_list:
            if f not in X.columns:
                X[f] = 0.0
        X = X[expected_feature_list]

    return X, y, df


def main():
    print("=" * 80)
    print("GeoFlare -- Punjab Pipeline & Cross-Region Model Transfer Evaluation")
    print("=" * 80)

    # 1. Compare class distributions side-by-side
    print("\n" + "=" * 80)
    print("1. SIDE-BY-SIDE CLASS DISTRIBUTION COMPARISON: KORBA vs. PUNJAB")
    print("=" * 80)

    df_k = pd.read_csv(KORBA_CSV)
    df_p = pd.read_csv(PUNJAB_CSV)

    k_counts = df_k["class_label"].value_counts()
    p_counts = df_p["class_label"].value_counts()
    all_classes = sorted(list(set(k_counts.index).union(set(p_counts.index))))

    print(f"{'Class Label':25s} | {'Korba Count':>12s} {'(%):':>7s} | {'Punjab Count':>12s} {'(%):':>7s}")
    print("-" * 75)
    for c in all_classes:
        kc = k_counts.get(c, 0)
        kpct = (kc / len(df_k)) * 100
        pc = p_counts.get(c, 0)
        ppct = (pc / len(df_p)) * 100
        print(f"{c:25s} | {kc:12,d} {kpct:6.1f}% | {pc:12,d} {ppct:6.1f}%")
    print("-" * 75)
    print(f"{'TOTAL':25s} | {len(df_k):12,d} {'100.0%':>7s} | {len(df_p):12,d} {'100.0%':>7s}")

    ag_k = k_counts.get("Agricultural Burning", 0)
    ag_p = p_counts.get("Agricultural Burning", 0)
    print(f"\nObservation on Agricultural Burning:")
    print(f"  - Korba:  {ag_k:,} rows ({ag_k/len(df_k)*100:.1f}%) -- primarily an industrial/mining belt")
    print(f"  - Punjab: {ag_p:,} rows ({ag_p/len(df_p)*100:.1f}%) -- massive stubble-burning volume!")

    # 2. Preprocess Punjab features
    print("\n" + "=" * 80)
    print("2. PREPROCESSING PUNJAB FEATURE TABLE")
    print("=" * 80)

    # Load expected feature list from Korba
    with open(KORBA_FEATURES, "r") as f:
        expected_features = [line.strip() for line in f if line.strip()]

    X_punjab, y_punjab, _ = load_and_preprocess(PUNJAB_CSV, expected_features)
    print(f"  Punjab feature matrix shape: {X_punjab.shape}")
    print(f"  Features match Korba schema exactly: {list(X_punjab.columns) == expected_features}")

    # 3. Train on Punjab (80/20 Stratified Split)
    print("\n" + "=" * 80)
    print("3. TRAINING & EVALUATING ON PUNJAB DATASET")
    print("=" * 80)

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_punjab, y_punjab, test_size=0.20, stratify=y_punjab, random_state=42
    )

    print(f"  Punjab Train set: {len(X_tr_p):,} rows | Test set: {len(X_te_p):,} rows")

    le_p = LabelEncoder()
    le_p.fit(y_punjab)
    y_tr_p_enc = le_p.transform(y_tr_p)
    y_te_p_enc = le_p.transform(y_te_p)

    # Train XGBoost on Punjab
    weights_p = compute_sample_weight("balanced", y_tr_p)
    xgb_p = XGBClassifier(
        n_estimators=300,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_p.fit(X_tr_p, y_tr_p_enc, sample_weight=weights_p)

    preds_p_enc = xgb_p.predict(X_te_p)
    preds_p = le_p.inverse_transform(preds_p_enc)

    print("\n  Punjab Model -- Test Set Classification Report:")
    print("  " + "-" * 65)
    print(classification_report(y_te_p, preds_p, digits=4))

    joblib.dump(xgb_p, PUNJAB_MODEL)
    print(f"  Saved Punjab model to: {PUNJAB_MODEL}")

    # 4. CROSS-REGION EVALUATION: KORBA MODEL ON PUNJAB DATA
    print("\n" + "=" * 80)
    print("4. CROSS-REGION EVALUATION: KORBA-TRAINED MODEL TESTED ON PUNJAB DATA")
    print("=" * 80)

    korba_model = joblib.load(KORBA_MODEL)
    print(f"  Loaded Korba classifier: {KORBA_MODEL}")

    # Label encoder for Korba model
    # Korba classes: ['Agricultural Burning', 'Industrial', 'Natural Fire', 'Unknown']
    le_k = LabelEncoder()
    le_k.fit(df_k["class_label"])

    # Predict entire Punjab dataset using Korba model
    korba_preds_enc = korba_model.predict(X_punjab)
    korba_preds = le_k.inverse_transform(korba_preds_enc)

    # Also compute on Punjab test set only for direct comparability
    korba_test_preds_enc = korba_model.predict(X_te_p)
    korba_test_preds = le_k.inverse_transform(korba_test_preds_enc)

    print("\n  --- A. Evaluation on Full Punjab Dataset (1,350 detections) ---")
    acc_full = accuracy_score(y_punjab, korba_preds)
    macro_f1_full = f1_score(y_punjab, korba_preds, average="macro")
    print(f"  Accuracy on Full Punjab: {acc_full * 100:.2f}%")
    print(f"  Macro F1 on Full Punjab: {macro_f1_full:.4f}\n")

    eval_classes = sorted(list(set(y_punjab).union(set(korba_preds))))
    print("  Classification Report (Korba model -> Full Punjab Data):")
    print("  " + "-" * 65)
    print(classification_report(y_punjab, korba_preds, digits=4, zero_division=0))

    print("  Confusion Matrix (Rows = True Punjab Class, Columns = Korba Model Prediction):")
    print(f"  Classes: {eval_classes}")
    cm_full = confusion_matrix(y_punjab, korba_preds, labels=eval_classes)
    for i, row in enumerate(cm_full):
        print(f"    {eval_classes[i]:25s} : {row}")

    print("\n  --- B. Evaluation on Punjab Held-Out Test Set (270 detections) ---")
    acc_test = accuracy_score(y_te_p, korba_test_preds)
    macro_f1_test = f1_score(y_te_p, korba_test_preds, average="macro")
    print(f"  Accuracy on Test Set: {acc_test * 100:.2f}%")
    print(f"  Macro F1 on Test Set: {macro_f1_test:.4f}\n")

    print("  Confusion Matrix (Rows = True Class, Columns = Predicted):")
    cm_test = confusion_matrix(y_te_p, korba_test_preds, labels=eval_classes)
    for i, row in enumerate(cm_test):
        print(f"    {eval_classes[i]:25s} : {row}")

    print("\n" + "=" * 80)
    print("CROSS-REGION EVALUATION SUMMARY:")
    print("=" * 80)
    ag_true_total = (y_punjab == "Agricultural Burning").sum()
    ag_pred_correct = ((y_punjab == "Agricultural Burning") & (korba_preds == "Agricultural Burning")).sum()
    print(f"  - Punjab True Agricultural Burning: {ag_true_total:,}")
    print(f"  - Correctly Identified by Korba Model: {ag_pred_correct:,} ({ag_pred_correct/ag_true_total*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
