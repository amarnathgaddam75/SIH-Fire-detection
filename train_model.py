"""
train_model.py  --  Train & Compare Classification Models
=========================================================
This script trains two machine-learning classifiers to predict the
`class_label` of satellite fire hotspots:

    1. RandomForestClassifier  (ensemble of decision trees)
    2. XGBClassifier           (gradient-boosted decision trees)

Both are trained on the same 80/20 stratified split and evaluated with
the same metrics, so the comparison is apples-to-apples.

Input:   data/feature_vector.csv   (from build_feature_table.py)
Output:  data/classifier.pkl       (best model, saved with joblib)

Dependencies:
    pip install pandas numpy scikit-learn xgboost joblib
"""

# ==========================================================================
# 0. IMPORTS
# ==========================================================================
# We import libraries one-by-one so you can see what each is used for.

import os
import warnings

import numpy as np                          # numerical operations
import pandas as pd                         # dataframe manipulation
import joblib                               # save/load Python objects to disk

# scikit-learn modules -- each one serves a specific purpose:
from sklearn.model_selection import (
    train_test_split,                       # split data into train/test
    StratifiedKFold,                        # k-fold splits that respect class ratios
    cross_val_score,                        # run cross-validation in one call
)
from sklearn.ensemble import RandomForestClassifier   # our first model
from sklearn.metrics import (
    classification_report,                  # per-class precision/recall/F1
    confusion_matrix,                       # NxN matrix of predictions vs truth
    f1_score,                               # single-number F1 metric
)

from xgboost import XGBClassifier           # our second model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")


# ==========================================================================
# STEP 1: LOAD THE CSV AND PRINT CLASS DISTRIBUTION
# ==========================================================================
# Before we do anything, we need to *see* the data.  How many examples of
# each class do we have?  Machine-learning models struggle when one class
# has 1,000 examples and another has only 10 -- that's called "class
# imbalance".  Printing the distribution up front lets us plan how to
# handle it.

print("=" * 72)
print("STEP 1: Load data/feature_vector.csv and inspect class balance")
print("=" * 72)

input_path = os.path.join(DATA_DIR, "feature_vector.csv")
df = pd.read_csv(input_path)

print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns\n")

# Value counts show how many rows belong to each class.
class_counts = df["class_label"].value_counts()
print("  class_label distribution:")
print("  " + "-" * 45)
for label, count in class_counts.items():
    pct = count / len(df) * 100
    print(f"    {label:25s} : {count:5,}  ({pct:5.1f}%)")
print("  " + "-" * 45)
print(f"    {'TOTAL':25s} : {len(df):5,}\n")

# Observation: "Unknown" is likely a tiny minority class.  We'll use
# class_weight="balanced" in Random Forest and examine Unknown recall
# specifically in Step 9.


# ==========================================================================
# STEP 2: SELECT AND ENCODE INPUT FEATURES
# ==========================================================================
# We pick the features the user specified.  A few need special handling:
#
#   - "brightness_temperature" -> the CSV column is called "brightness"
#     (this is the VIIRS/MODIS brightness temperature in Kelvin).
#
#   - "confidence" -> VIIRS uses string labels: "l" (low), "n" (nominal),
#     "h" (high).  We map these to 0, 1, 2 -- an ordinal encoding,
#     because there is a natural ordering (low < nominal < high).
#
#   - "daynight" -> binary: "D" (day) = 1, "N" (night) = 0.
#
#   - "ndvi", "ndbi", "ndwi" -> currently all "unavailable" (strings).
#     We replace "unavailable" with NaN so the imputer can handle them.

print("=" * 72)
print("STEP 2: Select and encode input features")
print("=" * 72)

# Map the user-requested feature names to actual CSV column names.
# "brightness_temperature" doesn't exist in the CSV; "brightness" does.
FEATURE_COLUMNS = [
    "frp",                          # Fire Radiative Power (MW)
    "brightness",                   # brightness temperature (K) -- user called it "brightness_temperature"
    "confidence",                   # detection confidence (l/n/h -> 0/1/2)
    "daynight",                     # day or night pass (D/N -> 1/0)
    "distance_to_industrial_m",     # metres to nearest industrial polygon
    "distance_to_farmland_m",       # metres to nearest farmland polygon
    "distinct_days",                # how many unique dates this cluster was detected
    "historical_mean_frp",          # cluster's average FRP
    "historical_max_frp",           # cluster's peak FRP
    "historical_std_frp",           # cluster's FRP variability
    "frp_deviation",                # this row's FRP / cluster mean FRP
    "night_detection_ratio",        # fraction of cluster detections at night
    "ndvi",                         # vegetation index (from Sentinel-2, or NaN)
    "ndbi",                         # built-up index
    "ndwi",                         # water index
]

# --- Encode categorical features before selecting columns ----------------

# daynight: "D" -> 1, "N" -> 0.
# Why 1 for day?  Arbitrary -- the model doesn't care about direction,
# only that day and night are distinguishable.
df["daynight"] = df["daynight"].map({"D": 1, "N": 0}).astype(float)

# confidence: ordinal encoding  l=0, n=1, h=2
# This preserves the ordering: low < nominal < high.
df["confidence"] = df["confidence"].map({"l": 0, "n": 1, "h": 2}).astype(float)

# ndvi, ndbi, ndwi: replace "unavailable" with NaN.
# pd.to_numeric with errors="coerce" turns anything non-numeric into NaN.
for col in ["ndvi", "ndbi", "ndwi"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Now select only the feature columns and the target.
X = df[FEATURE_COLUMNS].copy()
y = df["class_label"].copy()

print(f"  Feature matrix shape: {X.shape}  (rows x features)")
print(f"  Target vector shape:  {y.shape}")
print(f"  Features: {list(X.columns)}")
print()


# ==========================================================================
# STEP 3: IMPUTE MISSING VALUES + ADD "WAS MISSING" FLAGS
# ==========================================================================
# Some columns have NaN values:
#   - historical_mean_frp, historical_max_frp, historical_std_frp,
#     frp_deviation, night_detection_ratio  ->  NaN for noise points
#     (cluster_id == -1).
#   - ndvi, ndbi, ndwi  ->  NaN (no Sentinel-2 data available yet).
#
# Strategy:
#   1. For each column with any NaN, create a new binary column
#      "<feature>_was_missing" (1 if the value was NaN, 0 otherwise).
#      This lets the model *learn from missingness* -- e.g., noise points
#      (cluster_id == -1) might behave differently from clustered ones.
#
#   2. Fill NaN with the column median.  Median is more robust than mean
#      because it isn't affected by extreme outliers (e.g., a single
#      enormous FRP value).
#
# Why not just drop rows with NaN?  Because 465 noise points would be
# lost -- that's 23% of the data, and some might be "Unknown" class
# rows that we critically need for training.

print("=" * 72)
print("STEP 3: Impute missing values and add _was_missing flags")
print("=" * 72)

# Find columns that actually have missing values.
cols_with_missing = X.columns[X.isnull().any()].tolist()
print(f"  Columns with missing values: {cols_with_missing}")

for col in cols_with_missing:
    n_missing = X[col].isnull().sum()
    pct_missing = n_missing / len(X) * 100

    # Step 3a: Create the binary flag *before* imputing.
    flag_col = f"{col}_was_missing"
    X[flag_col] = X[col].isnull().astype(int)

    # Step 3b: Impute with the column median.
    # Special case: if ALL values are NaN (e.g., ndvi when no Sentinel-2
    # data exists), the median is NaN too.  In that case, fill with 0.0
    # -- the _was_missing flag will carry the real signal.
    median_val = X[col].median()
    if pd.isna(median_val):
        median_val = 0.0
        fill_note = "(all NaN -> filled with 0.0)"
    else:
        fill_note = f"filled with median={median_val:.4f}"
    X[col] = X[col].fillna(median_val)

    print(f"    {col}: {n_missing:,} NaN ({pct_missing:.1f}%) -> "
          f"{fill_note}, added {flag_col}")

print(f"\n  Final feature matrix shape: {X.shape}  "
      f"(added {len(cols_with_missing)} _was_missing columns)")
print()


# ==========================================================================
# STEP 4: TRAIN/TEST SPLIT (80/20, STRATIFIED)
# ==========================================================================
# We hold out 20% of the data as a "test set" that the model never sees
# during training.  This gives us an honest estimate of how the model
# will perform on new, unseen hotspots.
#
# "Stratified" means each split has the same class proportions as the
# full dataset.  Without stratification, the small "Unknown" class might
# end up entirely in the training set (or entirely in the test set),
# giving misleading results.
#
# random_state=42 makes the split reproducible -- you'll get the exact
# same train/test rows every time you run the script.

print("=" * 72)
print("STEP 4: Train/test split (80/20, stratified)")
print("=" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,           # 20% for testing
    stratify=y,               # preserve class proportions
    random_state=42,          # reproducibility
)

print(f"  Training set: {X_train.shape[0]:,} rows")
print(f"  Test set:     {X_test.shape[0]:,} rows")
print(f"\n  Training class distribution:")
for label, count in y_train.value_counts().items():
    print(f"    {label:25s} : {count:,}")
print(f"\n  Test class distribution:")
for label, count in y_test.value_counts().items():
    print(f"    {label:25s} : {count:,}")
print()


# ==========================================================================
# STEP 5: 5-FOLD CROSS-VALIDATION WITH RANDOM FOREST
# ==========================================================================
# Before we evaluate on the test set, we use cross-validation (CV) on
# the *training set* to get a more reliable performance estimate.
#
# How 5-fold CV works:
#   1. Split the training set into 5 equal "folds".
#   2. For each fold: train on 4 folds, evaluate on the remaining 1.
#   3. Average the 5 scores -> more stable than a single train/test split.
#
# We use "macro F1" as the metric.  Macro F1 computes F1 for each class
# independently and then averages them, giving equal weight to every
# class regardless of size.  This is important because our "Unknown"
# class is tiny -- accuracy would look great just by ignoring it.
#
# class_weight="balanced" tells the Random Forest to automatically
# upweight minority classes during training.  It's like artificially
# duplicating rare-class samples so the model pays attention to them.

print("=" * 72)
print("STEP 5: 5-fold stratified cross-validation (Random Forest)")
print("=" * 72)

rf = RandomForestClassifier(
    n_estimators=300,              # 300 decision trees in the forest
    class_weight="balanced",       # upweight minority classes
    random_state=42,               # reproducibility
    n_jobs=-1,                     # use all CPU cores for speed
)

# StratifiedKFold ensures each fold has the same class proportions.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score trains and evaluates the model 5 times automatically.
# scoring="f1_macro" computes macro-averaged F1 on each fold.
cv_scores = cross_val_score(
    rf, X_train, y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
)

print(f"  Cross-validated macro F1 scores: {cv_scores.round(4)}")
print(f"  Mean:  {cv_scores.mean():.4f}")
print(f"  Std:   {cv_scores.std():.4f}")
print()


# ==========================================================================
# STEP 6: TRAIN FINAL RANDOM FOREST AND EVALUATE ON TEST SET
# ==========================================================================
# Now we train the Random Forest on the *full* training set (not just
# 4/5 of it like in CV) and evaluate on the held-out test set.
#
# classification_report shows per-class precision, recall, and F1:
#   - Precision: of all rows predicted as class X, what fraction truly are X?
#   - Recall:    of all rows that truly are X, what fraction did we find?
#   - F1:        harmonic mean of precision and recall (balances both).
#
# confusion_matrix is an NxN grid where entry [i, j] = number of rows
# whose true class is i but were predicted as j.  The diagonal shows
# correct predictions; off-diagonal shows mistakes.

print("=" * 72)
print("STEP 6: Train final Random Forest -> test-set evaluation")
print("=" * 72)

# Fit on full training data.
rf.fit(X_train, y_train)

# Predict on the test set.
rf_preds = rf.predict(X_test)

# Get the list of unique class labels in a consistent order.
class_labels = sorted(y.unique())

# Macro F1 for later comparison with XGBoost.
rf_macro_f1 = f1_score(y_test, rf_preds, average="macro")

print(f"\n  Random Forest -- Test Set Classification Report:")
print("  " + "-" * 65)
print(classification_report(y_test, rf_preds, target_names=class_labels, digits=4))

print(f"  Macro F1 on test set: {rf_macro_f1:.4f}\n")

print(f"  Confusion Matrix (rows=true, cols=predicted):")
print(f"  Classes: {class_labels}")
cm_rf = confusion_matrix(y_test, rf_preds, labels=class_labels)
# Print with nice alignment.
for i, row in enumerate(cm_rf):
    print(f"    {class_labels[i]:25s} : {row}")
print()


# ==========================================================================
# STEP 7: TRAIN XGBOOST AND EVALUATE ON THE SAME TEST SET
# ==========================================================================
# XGBoost (eXtreme Gradient Boosting) builds trees sequentially: each
# new tree tries to correct the mistakes of the previous ones.  This
# often outperforms Random Forest, especially on tabular data.
#
# Key differences from Random Forest:
#   - RF builds trees independently in parallel -> ensemble by voting.
#   - XGB builds trees sequentially -> each focuses on previous errors.
#
# eval_metric="mlogloss" tells XGBoost to use multiclass log-loss
# internally for optimization (this is the standard choice for multi-
# class problems).
#
# XGBoost doesn't have class_weight="balanced", but we can compute
# sample_weight to achieve the same effect.

print("=" * 72)
print("STEP 7: Train XGBoost -> test-set evaluation")
print("=" * 72)

# Compute sample weights to handle class imbalance.
# The idea: if class X has N_total / (n_classes * N_x) weight,
# rare classes get higher weights -- same formula sklearn uses
# for class_weight="balanced".
from sklearn.utils.class_weight import compute_sample_weight
sample_weights_train = compute_sample_weight("balanced", y_train)

# XGBoost requires integer-encoded labels, not strings.  We use
# sklearn's LabelEncoder to map class names to integers (0, 1, 2)
# and then inverse_transform predictions back to the original names.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)  # fit on all labels so the mapping is consistent

y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

xgb = XGBClassifier(
    n_estimators=300,              # same number of trees as RF for fair comparison
    eval_metric="mlogloss",        # multiclass log-loss
    use_label_encoder=False,       # suppress deprecation warning
    random_state=42,               # reproducibility
    n_jobs=-1,                     # use all CPU cores
    verbosity=0,                   # suppress training output
)

# Fit with encoded labels and sample weights to handle imbalance.
xgb.fit(X_train, y_train_encoded, sample_weight=sample_weights_train)

# Predict on the test set and decode back to string labels.
xgb_preds_encoded = xgb.predict(X_test)
xgb_preds = le.inverse_transform(xgb_preds_encoded)

xgb_macro_f1 = f1_score(y_test, xgb_preds, average="macro")

print(f"\n  XGBoost -- Test Set Classification Report:")
print("  " + "-" * 65)
print(classification_report(y_test, xgb_preds, target_names=class_labels, digits=4))

print(f"  Macro F1 on test set: {xgb_macro_f1:.4f}\n")

print(f"  Confusion Matrix (rows=true, cols=predicted):")
print(f"  Classes: {class_labels}")
cm_xgb = confusion_matrix(y_test, xgb_preds, labels=class_labels)
for i, row in enumerate(cm_xgb):
    print(f"    {class_labels[i]:25s} : {row}")
print()


# ==========================================================================
# STEP 8: RANDOM FOREST FEATURE IMPORTANCES
# ==========================================================================
# Random Forest computes "feature importance" by measuring how much
# each feature reduces impurity (Gini) across all trees.  Higher
# values = more important for classification.
#
# This tells us which measurements matter most -- e.g., is distance
# to industrial zones more predictive than FRP?

print("=" * 72)
print("STEP 8: Random Forest feature importances (sorted)")
print("=" * 72)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n  Feature                          Importance")
print("  " + "-" * 50)
for feat, imp in importances.items():
    bar = "#" * int(imp * 100)  # visual bar
    print(f"    {feat:35s} : {imp:.4f}  {bar}")
print()


# ==========================================================================
# STEP 9: "UNKNOWN" CLASS ANALYSIS
# ==========================================================================
# The "Unknown" class represents hotspots with conflicting evidence --
# they're the hardest to classify.  This step checks: among test-set
# rows where the TRUE label is "Unknown", how many did each model
# predict correctly vs. misclassify?
#
# If a model confidently says "Industrial" for what is truly "Unknown",
# that's dangerous -- we'd miss conflicting signals.  High recall on
# "Unknown" means the model correctly flags ambiguous cases.

print("=" * 72)
print("STEP 9: 'Unknown' class prediction analysis")
print("=" * 72)

# Find test rows where the true label is "Unknown".
unknown_mask = y_test == "Unknown"
n_unknown = unknown_mask.sum()

if n_unknown > 0:
    # Random Forest predictions for Unknown-class rows.
    rf_unknown_preds = rf_preds[unknown_mask]
    rf_unknown_correct = (rf_unknown_preds == "Unknown").sum()
    rf_unknown_recall = rf_unknown_correct / n_unknown

    # XGBoost predictions for Unknown-class rows.
    xgb_unknown_preds = xgb_preds[unknown_mask]
    xgb_unknown_correct = (xgb_unknown_preds == "Unknown").sum()
    xgb_unknown_recall = xgb_unknown_correct / n_unknown

    print(f"\n  Test-set rows with true class_label == 'Unknown': {n_unknown}")
    print()
    print(f"  {'Metric':40s} {'Random Forest':>15s} {'XGBoost':>15s}")
    print("  " + "-" * 72)
    print(f"  {'Correctly predicted as Unknown':40s} "
          f"{rf_unknown_correct:>15,} {xgb_unknown_correct:>15,}")
    print(f"  {'Misclassified into another class':40s} "
          f"{n_unknown - rf_unknown_correct:>15,} "
          f"{n_unknown - xgb_unknown_correct:>15,}")
    print(f"  {'Unknown recall (fraction correct)':40s} "
          f"{rf_unknown_recall:>15.4f} {xgb_unknown_recall:>15.4f}")

    # Show WHERE each model sent the misclassified Unknown rows.
    print(f"\n  Where did each model send the misclassified 'Unknown' rows?")
    print()

    print(f"  Random Forest misclassification breakdown:")
    rf_misclass = pd.Series(rf_unknown_preds[rf_unknown_preds != "Unknown"])
    if len(rf_misclass) > 0:
        for label, count in rf_misclass.value_counts().items():
            print(f"    -> predicted '{label}': {count}")
    else:
        print(f"    (none -- all correctly predicted as Unknown!)")

    print(f"\n  XGBoost misclassification breakdown:")
    xgb_misclass = pd.Series(xgb_unknown_preds[xgb_unknown_preds != "Unknown"])
    if len(xgb_misclass) > 0:
        for label, count in xgb_misclass.value_counts().items():
            print(f"    -> predicted '{label}': {count}")
    else:
        print(f"    (none -- all correctly predicted as Unknown!)")
else:
    print("  No 'Unknown' rows in the test set (stratification may have "
          "put them all in training).")

print()


# ==========================================================================
# STEP 10: SAVE THE BEST MODEL
# ==========================================================================
# We compare the two models on macro F1 (the fairest metric for
# imbalanced data) and save the winner to disk using joblib.
#
# joblib is preferred over pickle for scikit-learn/XGBoost models
# because it handles large numpy arrays more efficiently.

print("=" * 72)
print("STEP 10: Save the best model to data/classifier.pkl")
print("=" * 72)

output_model_path = os.path.join(DATA_DIR, "classifier.pkl")

print(f"\n  Random Forest macro F1: {rf_macro_f1:.4f}")
print(f"  XGBoost macro F1:      {xgb_macro_f1:.4f}")
print()

if rf_macro_f1 >= xgb_macro_f1:
    winner_name = "Random Forest"
    winner_model = rf
else:
    winner_name = "XGBoost"
    winner_model = xgb

joblib.dump(winner_model, output_model_path)

print(f"  Winner: {winner_name} (macro F1 = "
      f"{max(rf_macro_f1, xgb_macro_f1):.4f})")
print(f"  Saved to: {output_model_path}")
print(f"\n  Reason: {winner_name} achieved a higher (or equal) macro F1 score")
print(f"  on the held-out test set, meaning it is better at correctly")
print(f"  classifying *all* classes (including the rare 'Unknown' class),")
print(f"  not just the majority classes.")
print()

# Also save the feature list so we know which columns the model expects.
feature_list_path = os.path.join(DATA_DIR, "model_features.txt")
with open(feature_list_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")
print(f"  Feature list saved to: {feature_list_path}")
print(f"    ({len(X.columns)} features)")

print("\n" + "=" * 72)
print("DONE! Review the outputs above before proceeding to hyperparameter tuning.")
print("=" * 72)
